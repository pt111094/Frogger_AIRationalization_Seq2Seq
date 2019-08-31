using System;
using System.IO;

using Frogger.Logging;
using System.Numerics;
using Frogger.Enums;
using System.Collections;
using System.Collections.Generic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace cSharpTest
{
    public class ButtonPress {
        public string buttonName;
        public string timestamp;
    }

    class Running {
        public static void Main(string[] args)
        {
            Random rnd = new Random();
            //y is all negative bc that's how utity works.
            // will manipulate it in Screenshot.cs to make it look right
            int frogY = rnd.Next(-400, -30);
            int frogX = rnd.Next(18, 482);
            Vector2 frogPosition = new Vector2(frogX, frogY);
            CarState[] carStates = new CarState[7];
            Vector2[] logPositions = new Vector2[3];
            int numLives = rnd.Next(1, 4);
            Vector2[] savedFrogPositions = new Vector2[3 - numLives];
            bool topTunnelLight = (rnd.Next(0, 2) == 0);
            bool bottomTunnelLight = (rnd.Next(0, 2) == 0);

            //for saved frog positions
            for (int i = 0; i < savedFrogPositions.Length; i++) {
                int randomX = rnd.Next(18, 482);
                int randomY = rnd.Next(-400, -30);
                savedFrogPositions[i] = new Vector2(randomX, randomY);
            }

            //i just randomly make 5 normal cars and 2 trucks
            // TODO: make sure if there is a way to find the color of cars or it's just hardcoded 4 red and 3 blue
            for (int i = 0; i < 5; i++) {
                int randomNum = rnd.Next(0, 500);
                CarState curr = new CarState
                {
                    carType = CarType.Car
                };

                curr.position = new Vector2(randomNum, -(189 + i * 30));
                carStates[i] = curr;
                
            }
            for (int i = 5; i < 7; i++)
            {
                int randomNum = rnd.Next(0, 500);
                CarState curr = new CarState
                {
                    carType = CarType.Truck
                };

                curr.position = new Vector2(randomNum, -(189 + i * 30));
                carStates[i] = curr;

            }

            for (int i = 0; i < 3; i++) {
                int randomNum = rnd.Next(0, 500);
                logPositions[i] = new Vector2(randomNum, -(70 + 30 * i));
            }

            //this is a sample test case to test Screenshot.cs
            DisplayState displayState = new DisplayState();
            displayState.frogPosition = frogPosition;
            Console.WriteLine("correct:" + frogPosition);
            displayState.carStates = carStates;

            displayState.logPositions = logPositions;
            displayState.savedFrogPositions = savedFrogPositions;
            displayState.numLives = numLives;
            displayState.topTunnelLight = topTunnelLight;
            displayState.bottomTunnelLight = bottomTunnelLight;
            List<DisplayState> items;
            using (StreamReader file = File.OpenText("../bin/screenshotFireStore/log_file.json"))
            using (JsonTextReader reader = new JsonTextReader(file)) {
                JObject o2 = (JObject)JToken.ReadFrom(reader);
       
                items = JsonConvert.DeserializeObject<List<DisplayState>>(o2["displayStates"].ToString());

                for (int i = 0; i < items.Count; i++) {
                    ScreenShot screenshot = new ScreenShot(items[i]);
                    String filename = "Screenshot_" + i.ToString() + ".png";
                    screenshot.BitmapConstructorEx(filename);
                    //Console.WriteLine("after");
                }


            }

            //TODO: notice we also create screeenshots for next state
            List<DisplayState> nextitems;
            using (StreamReader file = File.OpenText("../bin/screenshotFireStore/log_fileNext.json"))
            using (JsonTextReader reader = new JsonTextReader(file))
            {
                JObject o2 = (JObject)JToken.ReadFrom(reader);

                nextitems = JsonConvert.DeserializeObject<List<DisplayState>>(o2["displayStates"].ToString());

                for (int i = 0; i < nextitems.Count; i++)
                {
                    ScreenShot screenshot = new ScreenShot(nextitems[i]);
                    String filename = "NextScreenshot_" + i.ToString() + ".png";
                    screenshot.BitmapConstructorEx(filename);
                }


            }


        }
    }


}