import { useCallback, useEffect, useRef, useState } from 'react';
import { snakeToCamel } from '../lib/camelCase';
import type { ServerEvent, StartGenerationRequest } from '../types';

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected';

export function useWebSocket(onEvent: (event: ServerEvent) => void) {
  const wsRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout>>(undefined);
  const mountedRef = useRef(true);
  const intentionallyClosedRef = useRef(false);
  const onEventRef = useRef(onEvent);
  const connectRef = useRef<() => void>(undefined);

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
        // Auto-reconnect after 3s, unless intentionally closed by user
        if (!intentionallyClosedRef.current) {
          reconnectTimeout.current = setTimeout(() => connectRef.current?.(), 3000);
        }
      }
    };

    ws.onerror = () => {
      // Suppress noisy logs — onclose will fire after this and handle reconnect
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    onEventRef.current = onEvent;
  });

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

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
    intentionallyClosedRef.current = false;
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

  const close = useCallback(() => {
    clearTimeout(reconnectTimeout.current);
    intentionallyClosedRef.current = true;
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setStatus('disconnected');
    }
  }, []);

  return { status, send, close };
}
