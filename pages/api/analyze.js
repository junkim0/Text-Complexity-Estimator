export default function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }
  
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  try {
    const { text } = req.body;
    
    if (!text || !text.trim()) {
      res.status(400).json({
        success: false,
        error: 'Please provide text to analyze.'
      });
      return;
    }
    
    const result = analyzeTextComplexity(text.trim());
    res.status(200).json(result);
    
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Text analysis error: ' + error.message
    });
  }
}

function analyzeTextComplexity(text) {
  // Word separation (supports both English and other languages)
  const words = text.match(/\b\w+\b/g) || [];
  
  if (words.length === 0) {
    return {
      success: false,
      error: 'No words found to analyze.'
    };
  }
  
  // Sentence separation
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  
  // Basic statistics calculation
  const totalWords = words.length;
  const avgWordLength = words.reduce((sum, word) => sum + word.length, 0) / totalWords;
  const avgSentenceLength = totalWords / Math.max(sentences.length, 1);
  
  // Archaic/literary word detection (Shakespeare style)
  const archaicWords = (text.match(/\b(thou|thee|thy|thine|art|doth|hath|ye|wherefore|whence|whither|hither|thither|ere|oft|nay|aye|forsooth|prithee|mayhap|perchance|methinks|betwixt|amongst|whilst|upon|unto|beneath|'tis|'twas|'twere|'twill|'gainst)\b/gi) || []).length;
  
  // Complex word detection (7+ characters)
  const complexWords = words.filter(word => word.length >= 7).length;
  
  // Complexity score calculation (0.1 ~ 1.0)
  let score = 0.2; // Base score
  
  // Average word length contribution (max 0.3)
  score += Math.min(avgWordLength / 15, 0.3);
  
  // Average sentence length contribution (max 0.2)
  score += Math.min(avgSentenceLength / 25, 0.2);
  
  // Archaic/literary words contribution (max 0.2)
  score += Math.min((archaicWords / totalWords) * 3, 0.2);
  
  // Complex words contribution (max 0.1)
  score += Math.min((complexWords / totalWords) * 2, 0.1);
  
  // Score range limitation
  score = Math.min(Math.max(score, 0.1), 1.0);
  
  // Level determination
  let level, description;
  
  if (score < 0.3) {
    level = "Beginner";
    description = "Simple and easy to understand text.";
  } else if (score < 0.5) {
    level = "Elementary";
    description = "Basic vocabulary and sentence structure.";
  } else if (score < 0.7) {
    level = "Intermediate";
    description = "Moderate complexity text.";
  } else if (score < 0.85) {
    level = "Advanced";
    description = "Complex language and structure.";
  } else {
    level = "Expert";
    description = "Very complex and professional text.";
  }
  
  return {
    success: true,
    score: score,
    level: level,
    description: description,
    details: {
      totalWords: totalWords,
      avgWordLength: avgWordLength,
      avgSentenceLength: avgSentenceLength,
      archaicWords: archaicWords,
      complexWords: complexWords,
      sentences: sentences.length
    }
  };
} 