import { useCallback } from 'react';
import { Layout } from './components/Layout';
import { PromptInput } from './components/PromptInput';
import { TokenTree } from './components/TokenTree';
import { TextOutput } from './components/TextOutput';
import { KPIDashboard } from './components/KPIDashboard';
import { useWebSocket } from './hooks/useWebSocket';
import { useSpecDecState } from './hooks/useSpecDecState';

function App() {
  const { state, handleEvent, startGeneration, stopGeneration } = useSpecDecState();
  const { status, send, close } = useWebSocket(handleEvent);

  const onGenerate = useCallback(
    (prompt: string, k: number, temperature: number, maxTokens: number) => {
      startGeneration();
      send({ prompt, k, temperature, maxTokens });
    },
    [startGeneration, send]
  );

  const onStop = useCallback(() => {
    close();
    stopGeneration();
  }, [close, stopGeneration]);

  return (
    <Layout
      promptPanel={
        <PromptInput
          onGenerate={onGenerate}
          onStop={onStop}
          isGenerating={state.isGenerating}
          isConnected={status === 'connected'}
        />
      }
      treePanel={<TokenTree roots={state.treeRoots} />}
      textPanel={
        <TextOutput
          tokens={state.tokens}
          isGenerating={state.isGenerating}
        />
      }
      kpiPanel={<KPIDashboard history={state.metricsHistory} />}
    />
  );
}

export default App;
