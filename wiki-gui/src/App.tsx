import React, { useState } from 'react';
import { PlagAPI } from './api';
import APIError from './api/error';
import Paragraph from './api/paragraph';
import { PlagForm, PlagConsole, Header } from './ui';

function App() {
  const [plags, setPlags] = useState<Paragraph[]>([]);
  const [error, setError] = useState("");
  const [submitted, setSubmitted] = useState(false);

  async function onSubmit(text: string) {
    setSubmitted(true);
    setPlags([]);
    setError("");
    await PlagAPI.detect(text)
      .then((paragraphs) => setPlags(paragraphs))
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
          <PlagConsole paragraphs={plags} error={error} loading={submitted} />
        </div>
      </div>
      <div>
        as
      </div>
    </div>

  );
}

export default App;
