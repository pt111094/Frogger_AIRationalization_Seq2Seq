using Frogger.Logging;
using System.Drawing;
using System.IO;
using System;
using System.Numerics;
using Frogger.Enums;
using System.Drawing.Imaging;
namespace cSharpTest {
    public class SymbRep {
        private const int STATE_HEIGHT = 14;
        private const int STATE_WIDTH = 17;
        private const int STATE_WIDTH_BOUND = STATE_WIDTH - 1;
        private const int SCALE_X = 25;
        private const int SCALE_Y = 30;
        private int[,] currentState = new int[STATE_HEIGHT, STATE_WIDTH];
        private int[,] ResetState(DisplayState displayState) {
            int[,] state = new int[STATE_HEIGHT, STATE_WIDTH];
            int i;
            int j;
            for (i = 0; i < STATE_HEIGHT; i++) {
                for (j = 0; j < STATE_WIDTH; j++) {
                    state[i, j] = (int)StateKey.EMPTY;
                }
            }
            for (i = 0; i < STATE_WIDTH; i++) {
                state[13, i] = (int)StateKey.START;
                int platformKey = (int)StateKey.PLATFORM;
                state[12, i] = platformKey;
                state[4, i] = platformKey;
            }
            int winKey = (int)StateKey.WIN;
            int hazardZoneKey = (int)StateKey.HAZARD_ZONE;
            int wallKey = (int)StateKey.WALL;
            int tunnelOnKey = (int)StateKey.TUNNEL_ON_KEY;
            for (i = 0; i < 17; i++) {
                for (j = 1; j < 4; j++) {
                    state[j, i] = hazardZoneKey;
                }
            }
            for (i = 0; i < 2; i++) {
                state[0, i] = hazardZoneKey;
            }
            for (i = 2; i < 5; i++) {
                state[0, i] = hazardZoneKey;
            }
            for (i = 5; i < 7; i++) {
                state[0, i] = winKey;
            }
            for (i = 7; i < 10; i++) {
                state[0, i] = hazardZoneKey;
            }
            for (i = 10; i < 12; i++) {
                state[0, i] = winKey;
            }
            for (i = 12; i < 15; i++) {
                state[0, i] = hazardZoneKey;
            }
            for (i = 15; i < 17; i++) {
                state[0, i] = hazardZoneKey;
            }
            for (i = 6; i < 13; i++) {
                if (displayState.topTunnelLight == false && displayState.bottomTunnelLight == false) {
                    state[8, i] = wallKey;
                }
                else {
                    state[8, i] = tunnelOnKey;
                }
            }
            foreach (FrogEntity frog in savedFrogs) {
                if (frog.gameObject.activeSelf) {
                    SaveFrogState(state, frog);
                }
            }
            return state;
        }

        public void SetState(DisplayState displayState) {
            int round;
            float check;
            currentState = ResetState();
            Vector2[] logPositions = displayState.logPositions;
            CarStates[] carStates = displayState.carStates;
            Vector2 frogPostion = displayState.frogPosition;
            foreach (CarState carState in carStates) {
                //CarEntity car = carController.Entity;
                float c = carState.position.X;
                float c2 = carState.position.Y;
                c -= STATE_WIDTH_BOUND;
                if (c < 0) {
                    round = (int)-c / SCALE_X;
                }
                else {
                    round = (int)c / SCALE_X;
                }
                check = (c - SCALE_X * round) / SCALE_X;
                if (check > MIDDLE_THRESHOLD) {
                    round += 1;
                }
                int y = (int)((c2 - SCALE_Y) / SCALE_Y);
                char carKey;
                if ((int)pos.Y == 189 || (int)pos.Y == 189 + 60 ||
                    (int)pos.Y == 189 + 120 || (int)pos.Y == 189 + 180) {
                    carKey = (int)StateKey.RED_CAR;
                }
                else {
                    carKey = (int)StateKey.BLUE_CAR;
                }
                for (int i = 0; i < car.dimensions.width / SCALE_X; i++) {
                    int x = Mathf.Min(round + i, STATE_WIDTH_BOUND);
                    if (x >= 0 && x < STATE_WIDTH) {
                        currentState[y, x] = carKey;
                    }
                }
            }

            // Set log position
            foreach (Vector2 log in logPositions) {
                float c = log.X;
                float c2 = log.Y;
                c -= STATE_WIDTH_BOUND;
                if (c < 0) {
                    round = (int)-c / SCALE_X;
                }
                else {
                    round = (int)c / SCALE_X;
                }
                check = (c - SCALE_X * round) / SCALE_X;
                char logKey = (char)StateKey.LOG;
                int y = (int)((c2 - SCALE_Y) / SCALE_Y);
                if (c < 0) {
                    int intC = (int)c;
                    if (intC < -110) {
                        continue;
                    }
                    round = intC / SCALE_X;
                    float roundCheck = round - (int)(intC / SCALE_X);
                    int x = intC / SCALE_X;
                    if (roundCheck > MIDDLE_THRESHOLD) {
                        x += 1;
                    }
                    for (int i = 0; i < 4; i++) {
                        currentState[y, Mathf.Max(x + i, 0)] = logKey;
                    }
                }
                else {
                    if (check > MIDDLE_THRESHOLD) {
                        round += 1;
                    }
                    for (int i = 0; i < 4; i++) {
                        currentState[y, Mathf.Min(round + i, STATE_WIDTH_BOUND)] = logKey;
                    }
                }
            }

            // set frog position

            SaveFrogState(currentState, frogPostion);

            using (var sw = new StreamWriter("outputText.txt")) {
                for (int i = 0; i < STATE_HEIGHT; i++) {
                    for (int j = 0; j < STATE_WIDTH; j++) {
                        sw.Write(currentState[i, j] + " ");
                    }
                    sw.Write("\n");
                }

                sw.Flush();
                sw.Close();
            }
            Environment.Exit(0);
        }

        private void SaveFrogState(char[,] state, Vector2 frog) {
            float f = frog.X;
            float f2 = frog.Y;
            f -= STATE_WIDTH_BOUND;
            int round;
            if (f < 0) {
                round = (int)-f / SCALE_X;
            }
            else {
                round = (int)f / SCALE_X;
            }
            float check;
            check = (f - SCALE_X * round) / SCALE_X;
            if (check >= MIDDLE_THRESHOLD) {
                round += 1;
            }
            state[(int)((f2 - SCALE_Y) / SCALE_Y), Mathf.Min(round, STATE_WIDTH_BOUND)] = (char)(int)StateKey.FROG;
        }
    }
}