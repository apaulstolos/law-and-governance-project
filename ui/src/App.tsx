import { useState } from 'react';
import axios from 'axios';

type Analysis = {
  comment_id?: string;
  summary: string;
  label: string;
  probability: number;
  generated_response: string;
};

function App() {
  const [comment, setComment] = useState('');
  const [priorResponse, setPriorResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Analysis | null>(null);

  const analyze = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await axios.post('/api/analyze', {
        items: [
          {
            comment,
            prior_response: priorResponse || null,
          },
        ],
      });
      setResult(response.data.results[0]);
    } catch (err) {
      setError('Failed to analyze comment. Ensure the API is running and model trained.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 900, margin: '0 auto', padding: '2rem' }}>
      <h1>Comment Intelligence Demo</h1>
      <p>Paste a public comment and optional prior agency response. The service will summarize, classify, and draft a response.</p>

      <label style={{ display: 'block', marginTop: '1rem' }}>Comment</label>
      <textarea
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        rows={8}
        style={{ width: '100%', padding: '1rem', fontSize: '1rem' }}
        placeholder="Enter comment text"
      />

      <label style={{ display: 'block', marginTop: '1rem' }}>Prior Response (optional)</label>
      <textarea
        value={priorResponse}
        onChange={(e) => setPriorResponse(e.target.value)}
        rows={4}
        style={{ width: '100%', padding: '1rem', fontSize: '1rem' }}
        placeholder="Paste previous agency reply if available"
      />

      <button
        onClick={analyze}
        disabled={loading || !comment.trim()}
        style={{
          marginTop: '1.5rem',
          padding: '0.75rem 1.5rem',
          fontSize: '1rem',
          backgroundColor: '#2563eb',
          color: '#fff',
          border: 'none',
          borderRadius: 8,
          cursor: loading || !comment.trim() ? 'not-allowed' : 'pointer',
        }}
      >
        {loading ? 'Analyzing…' : 'Analyze comment'}
      </button>

      {error && (
        <p style={{ color: 'red', marginTop: '1rem' }}>{error}</p>
      )}

      {result && (
        <div style={{ marginTop: '2rem', background: '#fff', padding: '1.5rem', borderRadius: 12, boxShadow: '0 5px 20px rgba(15,23,42,0.1)' }}>
          <h2>Results</h2>
          <p><strong>Summary:</strong> {result.summary}</p>
          <p>
            <strong>Classification:</strong> {result.label}
            {' '}({(result.probability * 100).toFixed(1)}%)
          </p>
          <p><strong>Draft Response:</strong></p>
          <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit', background: '#f8fafc', padding: '1rem', borderRadius: 8 }}>
            {result.generated_response}
          </pre>
        </div>
      )}
    </div>
  );
}

export default App;
