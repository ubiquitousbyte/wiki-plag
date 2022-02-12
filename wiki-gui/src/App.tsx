import React, { useState } from 'react';
import { PlagAPI } from './api';
import APIError from './api/error';
import Paragraph from './api/paragraph';
import { PlagForm, Spinner } from './ui';
import ParagraphList from './ui/paragraph-list';

function App() {
  const [plags, setPlags] = useState<Paragraph[]>([]);
  const [error, setError] = useState("");

  async function onSubmit(text: string) {
    setPlags([]);
    await PlagAPI.detect(text)
      .then((paragraphs) => setPlags(paragraphs))
      .catch((err: APIError) => setError(err.detail))
  }

  function loadOrShow() {
    if (plags.length > 0) {
      return <ParagraphList paragraphs={plags} />;
    }
    return <Spinner />;
  }

  return (
    <div className="grid grid-cols-6 h-full">
      <div className="col-start-1 col-end-4 p-4 border-r-2 border-slate-200">
        <PlagForm onSubmit={onSubmit} />
      </div>
      <div className="col-start-4 col-end-7 p-4">
        {loadOrShow()}
      </div>
    </div>
  );
}

export default App;
