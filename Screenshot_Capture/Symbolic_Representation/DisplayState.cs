using System;
//using UnityEngine;
using System.Numerics;

namespace Frogger.Logging {

    /// <summary>
    /// Represents a frame state to display in-game.
    /// </summary>
    [Serializable]
    struct DisplayState {
        
        /// <summary> The position of the frog. </summary>
        public Vector2 frogPosition;
        /// <summary> States for each car. </summary>
        public CarState[] carStates;
        /// <summary> Positions for each log. </summary>
        public Vector2[] logPositions;
        /// <summary> Positions for each saved frog. </summary>
        public Vector2[] savedFrogPositions;
        /// <summary> The number of lives the player has. </summary>
        public int numLives;
        /// <summary> Whether the top tunnel light is on. </summary>
        public bool topTunnelLight;
        /// <summary> Whether the bottom tunnel light is on. </summary>
        public bool bottomTunnelLight;
    }
}
