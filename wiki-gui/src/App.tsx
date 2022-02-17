import React, { useState } from 'react';
import { PlagAPI, PlagCandidate } from './api';
import APIError from './api/error';
import { PlagForm, PlagConsole, Header, Footer, SimilarityChart } from './ui';

function App() {
  const [plags, setPlags] = useState<PlagCandidate[]>([]);
  const [error, setError] = useState("");
  const [submitted, setSubmitted] = useState(false);

  async function onSubmit(text: string) {
    setSubmitted(true);
    setPlags([]);
    setError("");
    await PlagAPI.detect(text)
      .then((candidates) => setPlags(candidates))
      .catch((err: APIError) => setError(err.detail));
  }

  return (
    <div className="grid">
      <div className="border-b-2 border-slate-200">
        <Header />
      </div>
      <div className="grid grid-cols-8 border-b-2 border-slate-200">
        <div className="col-start-1 col-end-3 p-4 border-r-2 border-slate-200">
          <div className="h-full flex flex-col align-center justify-center text-center">
            <PlagForm onSubmit={onSubmit} />
          </div>
        </div>
        <div className="col-start-3 col-end-9 p-4">
          <PlagConsole paragraphs={plags.map(plag => plag.paragraph)} error={error} loading={submitted} />
        </div>
      </div>
      {plags.length > 0 && (
        <div className="m-auto">
          <SimilarityChart data={plags.map(p => { return { title: p.paragraph.document.title, similarity: p.similarity } })} />
        </div>
      )}
      <div className="fixed bottom-0 w-full">
        <Footer />
      </div>
    </div>
  );
}

export default App;
