using System;
//using UnityEngine;
using System.Numerics;
using Frogger.Enums;

namespace Frogger.Logging {

    /// <summary>
    /// A display state for a car.
    /// </summary>
    [Serializable]
    struct CarState {

        /// <summary> The position of the car. </summary>
        public Vector2 position;
        /// <summary> The current type of car. </summary>
        public CarType carType;
    }
}
