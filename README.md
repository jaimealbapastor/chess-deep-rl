# Chess Engine with Deep Reinforcement Learning  

This project was developed as part of a university course on Reinforcement Learning. Our goal was to enhance an existing chess engine project by incorporating Q-Learning into its deep reinforcement learning framework. The original project, which served as the foundation for this work, can be found [here](https://github.com/zjeffer/chess-deep-rl).  

## Overview  
The original project implemented a chess engine inspired by the principles of AlphaZero. It used a convolutional neural network (CNN) trained through Monte Carlo Tree Search (MCTS) to make decisions. In our project, we aimed to improve the engine by:  
- Integrating **Q-Learning** for value-based move selection.  
- Adding a reward system for capturing pieces and achieving checkmate.  

## Results  
After training the models for 5 iterations each, we conducted 20 games in stochastic mode:  
- **Original Method**: 7 wins.  
- **Q-Learning Implementation**: 8 wins.  
- **Draws**: 5 games.  

While the Q-Learning approach showed promising results, further training and testing would be required to draw more conclusive insights.  
