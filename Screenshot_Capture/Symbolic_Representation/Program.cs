using System.Drawing;
using System.IO;
using System;
using Frogger.Logging;
using System.Numerics;
using Frogger.Enums;
using System.Drawing.Imaging;

namespace cSharpTest
{
    class ScreenShot
    {
        private DisplayState displayState;
        private CarState[] carStates;
        private Vector2[] logPositions;
        private Vector2[] savedFrogPositions;
        private int numLives;
        private bool topTunnelLight;
        private bool bottomTunnelLight;


        ////side walk
        private int sidewalklWidth = 464;
        private int sidewalkHeight = 25;
        //score board
        private int scoreBoardlWidth = 500;
        private int scoreBoardHeight = 30;

        public ScreenShot(DisplayState displayState) {
            this.displayState = displayState;
            //Console.WriteLine(this.displayState.carStates[0].position);
        }

        public void BitmapConstructorEx(String filename)
        {

            int width = 500, height = 425;

            int[,] amn = new int[20, 14];
            //bitmap
            //Console.WriteLine(this.displayState.carStates[0].position);
            Bitmap bmp = new Bitmap(width, height);
            Graphics graph = Graphics.FromImage(bmp);
            SymbRep sr = new SymbRep();
            sr.SetState(this.displayState);
            //scroeboard
            Image scroeboard = Image.FromFile("./Pics/Environment/scoreboard.png");
            graph.DrawImage(scroeboard, 0, 0, scoreBoardlWidth, scoreBoardHeight);
            //draw title
            // Create font and brush.
            Font drawFont = new Font("Arial", 16);
            SolidBrush drawBrush = new SolidBrush(Color.Blue);
            // Create point for upper-left corner of drawing.
            PointF drawPoint = new PointF(0.0F, 0.0F);

            //street
            Image street = Image.FromFile("./Pics/Environment/street.png");
            //street's width is same as side walk
            graph.DrawImage(street, 18, 150, sidewalklWidth, 250);

            //water
            Image water = Image.FromFile("./Pics/Environment/water.png");
            //30 = 0+scroeboard.height
            //130 = topsidewalk.y (160) - 30 (which is scoreboard height)
            graph.DrawImage(water, 18, 30, sidewalklWidth, 130);

            //saveZone B and C
            Image safeZone = Image.FromFile("./Pics/Environment/sidewalk.png");
            //50 and  2nd 30 accroding to move around in unity
            // first 30 is bc it's right under scoreboard
            graph.DrawImage(safeZone, 150, 30, 50, 30);
            //draw safeZone C, 300 is bc utity says so
            graph.DrawImage(safeZone, 300, 30, 50, 30);

            //Background set Up
            //bottom yello sidewalk
            Image bottomSideWalk = Image.FromFile("./Pics/Environment/sidewalk.png");
            graph.DrawImage(bottomSideWalk, 18, 400, sidewalklWidth, sidewalkHeight);
            //top yellow sidewalk
            Image topSideWalk = Image.FromFile("./Pics/Environment/sidewalk.png");
            graph.DrawImage(topSideWalk, 18, 160, sidewalklWidth, sidewalkHeight);
            graph.DrawString("Frogger Screenshot", drawFont, drawBrush, drawPoint);

            //roadmarker
            Image roadMarker = Image.FromFile("./Pics/Environment/roadmarker.png");

            for (int j = 0; j < 6; j++) {
                for (int i = 0; i < 19; i++)
                {
                    graph.DrawImage(roadMarker, 18 + 25
                                    * i, 214 + j * 30, 25, 2);
                }
            }


            /// <summary>
            /// draw 7 cars
            /// </summary>
            carStates = this.displayState.carStates;
            Image redcar = Image.FromFile("./Pics/Entities/redcar.png");
            Image bluecar = Image.FromFile("./Pics/Entities/reverse_bluecar.png");
            Image redtruck = Image.FromFile("./Pics/Entities/redtruck.png");
            Image bluetruck = Image.FromFile("./Pics/Entities/reverse_bluetruck.png");

            for (int i = 0; i < carStates.Length; i++) {
                CarType type = carStates[i].carType;
                Vector2 pos = carStates[i].position;
                if (type == CarType.Car) {

                    if ((int)pos.Y == 189 || (int)pos.Y == 189 + 60 ||
                        (int)pos.Y == 189 + 120 || (int)pos.Y == 189 + 180) {
                        graph.DrawImage(redcar, pos.X + 25, pos.Y, 50, 25);
                        //Console.WriteLine("inside");
                    } else {
                        graph.DrawImage(bluecar, pos.X - 30, pos.Y, 50, 25);
                    }

                } else {
                    if ((int)pos.Y == 189 || (int)pos.Y == 189 + 60 ||
                        (int)pos.Y == 189 + 120 || (int)pos.Y == 189 + 180)
                    {
                        graph.DrawImage(redtruck, pos.X + 50, pos.Y, 100, 25);
                    }
                    else
                    {
                        graph.DrawImage(bluetruck, pos.X - 55, pos.Y, 100, 25);
                    }
                }
            }

            //tunnel
            Image tunnel = Image.FromFile("./Pics/Environment/tunnel.png");
            //150,25 accoridng to img info
            graph.DrawImage(tunnel, 180.0F, 278.7F, 150, 25);
            Image offlight = Image.FromFile("./Pics/UI/TunnelLight.png");
            Image onlight = Image.FromFile("./Pics/UI/onLight.png");
            topTunnelLight = this.displayState.topTunnelLight;
            bottomTunnelLight = this.displayState.bottomTunnelLight;
            if (topTunnelLight == true) {
                graph.DrawImage(onlight, 200.0F, 278.7F, 8, 8);
            } else {
                graph.DrawImage(offlight, 200.0F, 278.7F, 8, 8);
            }
            if (bottomTunnelLight == true)
            {
                graph.DrawImage(onlight, 200.0F, 293.7F, 8, 8);
            }
            else
            {
                graph.DrawImage(offlight, 200.0F, 293.7F, 8, 8);
            }

            logPositions = this.displayState.logPositions;
            Image log = Image.FromFile("./Pics/Entities/log.png");
            for (int i = 0; i < 3; i++) {
                Vector2 pos = logPositions[i];
                if ((int)pos.Y == 70) {
                    graph.DrawImage(log, pos.X, pos.Y, 120, 25);
                }
                else {
                    graph.DrawImage(log, pos.X, pos.Y, 100, 25);
                }
            }

            //draw current frog
            Image frog = Image.FromFile("./Pics/Entities/frog.png");
            Vector2 frogPostion = this.displayState.frogPosition;
            graph.DrawImage(frog, frogPostion.X, frogPostion.Y, 25, 25);

            Image prefrog = Image.FromFile("./Pics/Entities/pre_frog.png");
            savedFrogPositions = this.displayState.savedFrogPositions;
            for (int i = 0; i < savedFrogPositions.Length; i++) {
                graph.DrawImage(prefrog, savedFrogPositions[i].X, savedFrogPositions[i].Y, 25, 25);
            }
            numLives = this.displayState.numLives;
            graph.DrawString(numLives.ToString() + "  lives left", drawFont, drawBrush, new PointF(200.0F, 0.0F));

            bmp.Save("../screenshotFireStore/" + filename);

        }
    }
}
