import { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [text, setText] = useState('');
  const [currentLanguage, setCurrentLanguage] = useState('ta');
  const [originalText, setOriginalText] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setText('');
    setOriginalText('');
    setCurrentLanguage('ta');
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please select a video file');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('video', file);

    try {
      const response = await axios.post('http://localhost:3000/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setText(response.data.text);
      setOriginalText(response.data.text);
      setCurrentLanguage('ta');
      alert('Text extracted successfully!');
      
    } catch (error) {
      console.error('Error:', error);
      alert('Error: ' + (error.response?.data?.details || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleLanguageChange = async (targetLang) => {
    if (!text || targetLang === currentLanguage) return;

    setLoading(true);

    try {
      const response = await axios.post('http://localhost:3000/api/translate', {
        text: originalText,
        fromLang: 'ta',
        toLang: targetLang
      });

      setText(response.data.text);
      setCurrentLanguage(targetLang);
      
    } catch (error) {
      console.error('Translation error:', error);
      alert('Translation failed: ' + (error.response?.data?.details || error.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="container">
        <header>
          <h1>🎥 Video Text Extractor</h1>
          <p className="subtitle">Extract and translate text from Tamil videos</p>
        </header>
        
        <div className="upload-section">
          <input 
            type="file" 
            accept="video/*" 
            onChange={handleFileChange}
            disabled={loading}
          />
          <button 
            onClick={handleUpload} 
            disabled={!file || loading}
            className="btn-primary"
          >
            {loading ? 'Processing...' : 'Extract Text'}
          </button>
        </div>

        {text && (
          <div className="result-section">
            <div className="language-selector">
              <label>Select Language:</label>
              <select 
                value={currentLanguage} 
                onChange={(e) => handleLanguageChange(e.target.value)}
                disabled={loading}
              >
                <option value="ta">Tamil</option>
                <option value="en">English</option>
                <option value="hi">Hindi</option>
                <option value="es">Spanish</option>
                <option value="fr">French</option>
                <option value="de">German</option>
              </select>
            </div>

            <div className="text-display">
              <h3>Extracted Text:</h3>
              <textarea 
                value={text}
                readOnly
                rows={10}
              />
            </div>
          </div>
        )}

        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Processing...</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
