import { useCallback, useEffect, useRef, useState } from 'react';
import type { ServerEvent, StartGenerationRequest } from '../types';

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected';

// Convert snake_case keys to camelCase recursively
function snakeToCamel(obj: unknown): unknown {
  if (Array.isArray(obj)) {
    return obj.map(snakeToCamel);
  }
  if (obj !== null && typeof obj === 'object') {
    return Object.fromEntries(
      Object.entries(obj as Record<string, unknown>).map(([key, val]) => [
        key.replace(/_([a-z])/g, (_, c) => c.toUpperCase()),
        snakeToCamel(val),
      ])
    );
  }
  return obj;
}

export function useWebSocket(onEvent: (event: ServerEvent) => void) {
  const wsRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout>>(undefined);
  const mountedRef = useRef(true);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  const connect = useCallback(() => {
    // Don't connect if unmounted or already open/connecting
    if (!mountedRef.current) return;
    if (
      wsRef.current?.readyState === WebSocket.OPEN ||
      wsRef.current?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }

    setStatus('connecting');
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/tokens`);

    ws.onopen = () => {
      if (mountedRef.current) {
        setStatus('connected');
      }
    };

    ws.onmessage = (event) => {
      try {
        const raw = JSON.parse(event.data);
        const converted = snakeToCamel(raw) as ServerEvent;
        onEventRef.current(converted);
      } catch (e) {
        console.error('Failed to parse WS message:', e);
      }
    };

    ws.onclose = () => {
      if (mountedRef.current) {
        setStatus('disconnected');
        wsRef.current = null;
        // Auto-reconnect after 3s
        reconnectTimeout.current = setTimeout(connect, 3000);
      }
    };

    ws.onerror = () => {
      // Suppress noisy logs — onclose will fire after this and handle reconnect
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      clearTimeout(reconnectTimeout.current);
      if (wsRef.current) {
        wsRef.current.onclose = null; // prevent reconnect from cleanup
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  const send = useCallback((request: StartGenerationRequest) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected');
      return;
    }
    // Send as snake_case to match Python backend
    wsRef.current.send(
      JSON.stringify({
        prompt: request.prompt,
        max_tokens: request.maxTokens,
        temperature: request.temperature,
        k: request.k,
      })
    );
  }, []);

  return { status, send };
}
