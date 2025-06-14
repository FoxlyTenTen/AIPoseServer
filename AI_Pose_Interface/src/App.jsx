import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// Pose images mapping
const poseImages = {
  RAISE_HANDS: '/images/RAISE_HAND.jpg',
  SIU: '/images/SIU.jpg',
  JOG: '/images/JOG.jpg',
};

const PoseGameSimple = () => {
  const [gameState, setGameState] = useState({
    currentPose: '',
    currentScore: 0,
    timeLeft: 0,
    successCount: 0,
    failCount: 0,
    result: null,
    isPlaying: false
  });
  const [firstFrameSeen, setFirstFrameSeen] = useState(false);
  const audioRef = useRef(null);
  const videoRef = useRef(null);

  const startGame = async () => {
    await fetch('http://localhost:5000/start_game', { method: 'POST' });
    setFirstFrameSeen(false);
    setGameState(gs => ({ ...gs, isPlaying: true }));
  };

  const endGame = async () => {
    await fetch('http://localhost:5000/stop_game', { method: 'POST' });
    audioRef.current?.pause();
    audioRef.current.currentTime = 0;
    setGameState(gs => ({ ...gs, isPlaying: false }));
  };

  // Poll backend for game state
  useEffect(() => {
    if (!gameState.isPlaying) return;
    const interval = setInterval(async () => {
      const res = await fetch('http://localhost:5000/game_state');
      const data = await res.json();
      setGameState(gs => ({
        ...gs,
        currentPose: data.current_pose,
        currentScore: data.current_score,
        timeLeft: data.time_left,
        successCount: data.success_count,
        failCount: data.fail_count,
        result: data.result
      }));
    }, 200);
    return () => clearInterval(interval);
  }, [gameState.isPlaying]);

  // Compute current pose number
  const poseNumber = gameState.successCount + gameState.failCount + 1;

  return (
    <div className="app-container">
      <div className="game-container">
        {!gameState.isPlaying ? (
          <div className="start-screen">
            <h1>Pose Challenge</h1>
            <button className="start-button" onClick={startGame}>Start Game</button>
          </div>
        ) : (
          <>
            <audio ref={audioRef} src="/audio/subway.mp3" preload="auto" loop />

            <div className="game-grid">
              {/* Left Panel: Clue */}
              <div className="pose-clue">
                {gameState.currentPose && (
                  <img
                    src={poseImages[gameState.currentPose]}
                    alt={gameState.currentPose}
                  />
                )}
                <p>Follow this silly pose...</p>
              </div>

              {/* Center Panel: Countdown & Video */}
              <div className="video-container">
                <div className="game-stats">
                  <div className="stat-item">
                    <span className="stat-label">Pose:</span>
                    <span className="stat-value">{poseNumber}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Success:</span>
                    <span className="stat-value">{gameState.successCount}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Failed:</span>
                    <span className="stat-value">{gameState.failCount}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Score:</span>
                    <span className="stat-value">{gameState.currentScore.toFixed(3)}%</span>
                  </div>
                </div>
                <div className="countdown">
                  Get ready in {Math.ceil(gameState.timeLeft)}...
                </div>
                <div className="video-feed">
                  <img
                    ref={videoRef}
                    src="http://localhost:5000/video_feed"
                    alt="Video Feed"
                    onLoad={() => {
                      if (!firstFrameSeen) {
                        setFirstFrameSeen(true);
                        audioRef.current?.play().catch(() => { });
                      }
                    }}
                  />
                </div>
                {gameState.result && (
                  <div className={`result-banner ${gameState.result.toLowerCase()}`}>
                    {gameState.result === 'SUCCESS' ? 'Great Job! 🎉' : 'Try Again! 💪'}
                  </div>
                )}
              </div>

              {/* Right Panel: Audio hint */}
              <div className="audio-hint">
                <p>You'll hear a chime when it's time to hold a pose</p>
              </div>

              {/* End Game button */}
              <div className="end-game-button">
                <button onClick={endGame}>End Game</button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default PoseGameSimple;