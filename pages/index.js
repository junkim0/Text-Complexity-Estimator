import { useState } from 'react';
import Head from 'next/head';

export default function Home() {
  const [text, setText] = useState("To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles.");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const examples = {
    simple: "The cat sat on the mat. It was a sunny day. The cat was happy.",
    shakespeare: "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles.",
    academic: "The methodology employed in this comprehensive study utilizes sophisticated statistical analyses to examine the multifaceted relationships between various socioeconomic variables and their corresponding impacts on educational outcomes.",
    complex: "The epistemological implications of poststructuralist hermeneutics necessitate a comprehensive reconceptualization of ontological paradigms within contemporary phenomenological discourse."
  };

  const setExample = (type) => {
    setText(examples[type]);
    setResult(null);
    setError(null);
  };

  const analyzeText = async () => {
    if (!text.trim()) {
      alert('Please enter text to analyze.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text.trim() })
      });

      const data = await response.json();

      if (data.success) {
        setResult(data);
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError('Analysis error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setText('');
    setResult(null);
    setError(null);
  };

  return (
    <>
      <Head>
        <title>Text Complexity Estimator</title>
        <meta name="description" content="Analyze text complexity with advanced algorithms" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className="container">
        <h1>🔍 Text Complexity Estimator</h1>
        
        <div className="status">
          ✅ System is working perfectly!
        </div>
        
        <textarea 
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter text to analyze..."
          onKeyDown={(e) => {
            if (e.ctrlKey && e.key === 'Enter') {
              analyzeText();
            }
          }}
        />
        
        <div className="examples">
          <button className="example-btn" onClick={() => setExample('simple')}>Simple Text</button>
          <button className="example-btn" onClick={() => setExample('shakespeare')}>Shakespeare</button>
          <button className="example-btn" onClick={() => setExample('academic')}>Academic Text</button>
          <button className="example-btn" onClick={() => setExample('complex')}>Complex Text</button>
        </div>
        
        <br />
        <button onClick={analyzeText} disabled={loading}>
          {loading ? 'Analyzing...' : '🔍 Analyze Text'}
        </button>
        <button onClick={clearAll}>🗑️ Clear</button>
        
        {loading && (
          <div className="result">
            <div className="loading">Analyzing text...</div>
          </div>
        )}

        {error && (
          <div className="result">
            <div className="error">Error: {error}</div>
          </div>
        )}

        {result && !loading && (
          <div className="result">
            <div className="score">{result.score.toFixed(3)}</div>
            <div className="level">{result.level}</div>
            <div className="description">{result.description}</div>
            <div style={{ marginTop: '15px', fontSize: '0.9em', color: '#666' }}>
              <strong>Analysis Details:</strong><br />
              • Average word length: {result.details.avgWordLength.toFixed(1)} characters<br />
              • Average sentence length: {result.details.avgSentenceLength.toFixed(1)} words<br />
              • Archaic/literary words: {result.details.archaicWords}<br />
              • Total words: {result.details.totalWords}
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        .container {
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          max-width: 800px;
          margin: 0 auto;
          padding: 30px;
          background: white;
          border-radius: 15px;
          box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        body {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          min-height: 100vh;
          margin: 0;
          padding: 20px;
          color: #333;
        }

        h1 {
          text-align: center;
          color: #4a5568;
          margin-bottom: 30px;
          font-size: 2.5em;
        }

        .status {
          background: #48bb78;
          color: white;
          padding: 10px 20px;
          border-radius: 8px;
          text-align: center;
          margin-bottom: 20px;
          font-weight: bold;
        }

        textarea {
          width: 100%;
          min-height: 120px;
          padding: 15px;
          border: 2px solid #e2e8f0;
          border-radius: 8px;
          font-size: 16px;
          font-family: inherit;
          resize: vertical;
          margin-bottom: 20px;
          box-sizing: border-box;
        }

        textarea:focus {
          outline: none;
          border-color: #667eea;
          box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        button {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border: none;
          padding: 12px 30px;
          border-radius: 8px;
          font-size: 16px;
          font-weight: bold;
          cursor: pointer;
          transition: transform 0.2s;
          margin-right: 10px;
        }

        button:hover:not(:disabled) {
          transform: translateY(-2px);
        }

        button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
          transform: none;
        }

        .examples {
          margin-top: 20px;
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
        }

        .example-btn {
          background: #e2e8f0;
          color: #4a5568;
          border: none;
          padding: 8px 15px;
          border-radius: 20px;
          font-size: 14px;
          cursor: pointer;
          transition: background 0.2s;
        }

        .example-btn:hover {
          background: #cbd5e0;
          transform: none;
        }

        .result {
          margin-top: 30px;
          padding: 20px;
          background: #f7fafc;
          border-radius: 8px;
          border-left: 4px solid #667eea;
        }

        .score {
          font-size: 2em;
          font-weight: bold;
          color: #667eea;
          text-align: center;
          margin-bottom: 15px;
        }

        .level {
          font-size: 1.3em;
          font-weight: bold;
          text-align: center;
          margin-bottom: 10px;
        }

        .description {
          text-align: center;
          color: #666;
          font-size: 1.1em;
        }

        .error {
          background: #fed7d7;
          color: #c53030;
          padding: 15px;
          border-radius: 8px;
        }

        .loading {
          text-align: center;
          color: #666;
          font-style: italic;
        }
      `}</style>

      <style jsx global>{`
        body {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          min-height: 100vh;
          margin: 0;
          padding: 20px;
          color: #333;
        }
      `}</style>
    </>
  );
} 