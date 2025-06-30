export default function handler(req, res) {
  res.status(200).json({ 
    message: 'SUCCESS! Text Complexity Estimator is Working!',
    status: 'working',
    timestamp: new Date().toISOString()
  });
} 