export default function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }
  
  if (req.method === 'GET') {
    // Serve the web interface
    res.setHeader('Content-Type', 'text/html');
    res.status(200).send(`
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>텍스트 복잡도 분석기</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
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
        button:hover {
            transform: translateY(-2px);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
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
        .error {
            background: #fed7d7;
            color: #c53030;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .loading {
            text-align: center;
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 텍스트 복잡도 분석기</h1>
        
        <div class="status">
            ✅ 시스템이 정상적으로 작동 중입니다!
        </div>
        
        <textarea id="textInput" placeholder="분석할 텍스트를 입력하세요...">To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles.</textarea>
        
        <div class="examples">
            <button class="example-btn" onclick="setExample('simple')">간단한 텍스트</button>
            <button class="example-btn" onclick="setExample('shakespeare')">셰익스피어</button>
            <button class="example-btn" onclick="setExample('academic')">학술 텍스트</button>
            <button class="example-btn" onclick="setExample('korean')">한국어 텍스트</button>
        </div>
        
        <br>
        <button onclick="analyzeText()" id="analyzeBtn">🔍 텍스트 분석하기</button>
        <button onclick="clearAll()" id="clearBtn">🗑️ 초기화</button>
        
        <div id="result" style="display: none;"></div>
    </div>

    <script>
        const examples = {
            simple: "The cat sat on the mat. It was a sunny day. The cat was happy.",
            shakespeare: "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles.",
            academic: "The methodology employed in this comprehensive study utilizes sophisticated statistical analyses to examine the multifaceted relationships between various socioeconomic variables and their corresponding impacts on educational outcomes.",
            korean: "오늘은 날씨가 매우 좋습니다. 햇살이 따뜻하고 바람이 시원해서 산책하기에 완벽한 날입니다."
        };

        function setExample(type) {
            document.getElementById('textInput').value = examples[type];
        }

        function clearAll() {
            document.getElementById('textInput').value = '';
            document.getElementById('result').style.display = 'none';
        }

        async function analyzeText() {
            const text = document.getElementById('textInput').value.trim();
            const resultDiv = document.getElementById('result');
            const analyzeBtn = document.getElementById('analyzeBtn');
            
            if (!text) {
                alert('분석할 텍스트를 입력해주세요.');
                return;
            }
            
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = '분석 중...';
            
            resultDiv.innerHTML = '<div class="loading">텍스트를 분석하고 있습니다...</div>';
            resultDiv.style.display = 'block';
            
            try {
                const response = await fetch('/api', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    resultDiv.innerHTML = \`
                        <div class="score">\${data.score.toFixed(3)}</div>
                        <div class="level">\${data.level}</div>
                        <div class="description">\${data.description}</div>
                        <div style="margin-top: 15px; font-size: 0.9em; color: #666;">
                            <strong>분석 결과:</strong><br>
                            • 평균 단어 길이: \${data.details.avgWordLength.toFixed(1)}자<br>
                            • 평균 문장 길이: \${data.details.avgSentenceLength.toFixed(1)}단어<br>
                            • 고어/문어체 단어: \${data.details.archaicWords}개<br>
                            • 전체 단어 수: \${data.details.totalWords}개
                        </div>
                    \`;
                } else {
                    resultDiv.innerHTML = \`<div class="error">오류: \${data.error}</div>\`;
                }
            } catch (error) {
                resultDiv.innerHTML = \`<div class="error">분석 중 오류가 발생했습니다: \${error.message}</div>\`;
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = '🔍 텍스트 분석하기';
            }
        }
        
        // Enter key support
        document.getElementById('textInput').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                analyzeText();
            }
        });
    </script>
</body>
</html>
    `);
    return;
  }
  
  if (req.method === 'POST') {
    try {
      const { text } = req.body;
      
      if (!text || !text.trim()) {
        res.status(400).json({
          success: false,
          error: '분석할 텍스트를 입력해주세요.'
        });
        return;
      }
      
      const result = analyzeTextComplexity(text.trim());
      res.status(200).json(result);
      
    } catch (error) {
      res.status(500).json({
        success: false,
        error: '텍스트 분석 중 오류가 발생했습니다: ' + error.message
      });
    }
    return;
  }
  
  res.status(405).json({ error: '지원하지 않는 HTTP 메서드입니다.' });
}

function analyzeTextComplexity(text) {
  // 단어 분리 (영어와 한국어 모두 지원)
  const words = text.match(/\b\w+\b/g) || [];
  
  if (words.length === 0) {
    return {
      success: false,
      error: '분석할 단어를 찾을 수 없습니다.'
    };
  }
  
  // 문장 분리
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  
  // 기본 통계 계산
  const totalWords = words.length;
  const avgWordLength = words.reduce((sum, word) => sum + word.length, 0) / totalWords;
  const avgSentenceLength = totalWords / Math.max(sentences.length, 1);
  
  // 고어/문어체 단어 탐지 (셰익스피어 스타일)
  const archaicWords = (text.match(/\b(thou|thee|thy|thine|art|doth|hath|ye|wherefore|whence|whither|hither|thither|ere|oft|nay|aye|forsooth|prithee|mayhap|perchance|methinks|betwixt|amongst|whilst|upon|unto|beneath|'tis|'twas|'twere|'twill|'gainst)\b/gi) || []).length;
  
  // 복잡한 단어 탐지 (7글자 이상)
  const complexWords = words.filter(word => word.length >= 7).length;
  
  // 복잡도 점수 계산 (0.1 ~ 1.0)
  let score = 0.2; // 기본 점수
  
  // 평균 단어 길이 기여도 (최대 0.3)
  score += Math.min(avgWordLength / 15, 0.3);
  
  // 평균 문장 길이 기여도 (최대 0.2)
  score += Math.min(avgSentenceLength / 25, 0.2);
  
  // 고어/문어체 단어 기여도 (최대 0.2)
  score += Math.min((archaicWords / totalWords) * 3, 0.2);
  
  // 복잡한 단어 기여도 (최대 0.1)
  score += Math.min((complexWords / totalWords) * 2, 0.1);
  
  // 점수 범위 제한
  score = Math.min(Math.max(score, 0.1), 1.0);
  
  // 레벨 결정
  let level, description;
  
  if (score < 0.3) {
    level = "초급 (Beginner)";
    description = "간단하고 이해하기 쉬운 텍스트입니다.";
  } else if (score < 0.5) {
    level = "기초 (Elementary)";
    description = "기본적인 어휘와 문장 구조를 사용한 텍스트입니다.";
  } else if (score < 0.7) {
    level = "중급 (Intermediate)";
    description = "적당한 복잡도를 가진 텍스트입니다.";
  } else if (score < 0.85) {
    level = "고급 (Advanced)";
    description = "복잡한 언어와 구조를 사용한 텍스트입니다.";
  } else {
    level = "전문가 (Expert)";
    description = "매우 복잡하고 전문적인 텍스트입니다.";
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