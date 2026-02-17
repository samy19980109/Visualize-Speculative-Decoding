import { memo, useState } from 'react';

interface PromptInputProps {
  onGenerate: (prompt: string, k: number, temperature: number, maxTokens: number) => void;
  onStop: () => void;
  isGenerating: boolean;
  isConnected: boolean;
}

export const PromptInput = memo(function PromptInput({ onGenerate, onStop, isGenerating, isConnected }: PromptInputProps) {
  const [prompt, setPrompt] = useState('Explain how transformers work in machine learning.');
  const [k, setK] = useState(8);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(256);

  const handleSubmit = () => {
    if (!prompt.trim() || isGenerating) return;
    onGenerate(prompt, k, temperature, maxTokens);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, height: '100%' }}>
      <h2 style={{ fontSize: 14, fontWeight: 600, color: '#94a3b8', margin: 0, textTransform: 'uppercase', letterSpacing: 1 }}>
        Prompt
      </h2>

      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Enter your prompt..."
        rows={6}
        style={{
          flex: 1,
          minHeight: 120,
          backgroundColor: '#1e293b',
          border: '1px solid #334155',
          borderRadius: 8,
          color: '#e2e8f0',
          padding: 12,
          fontSize: 14,
          resize: 'vertical',
          fontFamily: 'inherit',
          outline: 'none',
        }}
      />

      {/* Controls */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <label style={{ fontSize: 12, color: '#94a3b8', minWidth: 80 }}>
            K (drafts)
          </label>
          <input
            type="range"
            min={1}
            max={16}
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: 12, color: '#e2e8f0', minWidth: 24 }}>{k}</span>
        </div>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <label style={{ fontSize: 12, color: '#94a3b8', minWidth: 80 }}>
            Temperature
          </label>
          <input
            type="range"
            min={0}
            max={2}
            step={0.1}
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: 12, color: '#e2e8f0', minWidth: 24 }}>{temperature}</span>
        </div>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <label style={{ fontSize: 12, color: '#94a3b8', minWidth: 80 }}>
            Max tokens
          </label>
          <input
            type="range"
            min={64}
            max={1024}
            step={64}
            value={maxTokens}
            onChange={(e) => setMaxTokens(Number(e.target.value))}
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: 12, color: '#e2e8f0', minWidth: 24 }}>{maxTokens}</span>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 8 }}>
        <button
          onClick={handleSubmit}
          disabled={isGenerating || !isConnected || !prompt.trim()}
          style={{
            flex: 1,
            padding: '10px 20px',
            backgroundColor: isGenerating ? '#334155' : '#6366f1',
            color: '#fff',
            border: 'none',
            borderRadius: 8,
            fontSize: 14,
            fontWeight: 600,
            cursor: isGenerating ? 'not-allowed' : 'pointer',
            opacity: (!isConnected || !prompt.trim()) ? 0.5 : 1,
            transition: 'background-color 200ms',
          }}
        >
          {isGenerating ? 'Generating...' : 'Generate'}
        </button>
        
        {isGenerating && (
          <button
            onClick={onStop}
            style={{
              padding: '10px 20px',
              backgroundColor: '#ef4444',
              color: '#fff',
              border: 'none',
              borderRadius: 8,
              fontSize: 14,
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'background-color 200ms',
            }}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#dc2626'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#ef4444'}
          >
            Stop
          </button>
        )}
      </div>

      {!isConnected && (
        <p style={{ fontSize: 12, color: '#f59e0b', margin: 0 }}>
          Connecting to backend...
        </p>
      )}
    </div>
  );
});
