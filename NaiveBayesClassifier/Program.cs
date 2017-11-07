using System;
using System.Data;
using System.Globalization;
using System.IO;
using System.Text;
using System.Collections.Generic;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Reflection;

namespace NaiveBayes
{
    public class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Choose your option\n1-first 3 features will be used\n2-all features will be used");

            int featureNumber = Convert.ToInt32(Console.ReadLine());

            DataTable tableAbalone = new DataTable();
            if (featureNumber == 1)
            {
                tableAbalone.Columns.Add("Class");
                tableAbalone.Columns.Add("Sex", typeof(double));
                tableAbalone.Columns.Add("Length", typeof(double));
                tableAbalone.Columns.Add("Diameter", typeof(double));

            }
            else if (featureNumber == 2)
            {
                tableAbalone.Columns.Add("Class");
                tableAbalone.Columns.Add("Sex", typeof(double));
                tableAbalone.Columns.Add("Length", typeof(double));
                tableAbalone.Columns.Add("Diameter", typeof(double));
                tableAbalone.Columns.Add("Height", typeof(double));
                tableAbalone.Columns.Add("WholeWeight", typeof(double));
                tableAbalone.Columns.Add("Shucked", typeof(double));
                tableAbalone.Columns.Add("Viscera", typeof(double));
                tableAbalone.Columns.Add("Shellweight", typeof(double));
            }
            else
            {
                Console.WriteLine("Wrong Number!!");
                return;
            }

            int count = 0;
            int numberOfTrainData = 0;
            while (count < 3)
            {
                if (count == 0)
                {
                    numberOfTrainData = 100;
                }
                else if (count == 1)
                {
                    numberOfTrainData = 1000;
                }
                else if (count == 2)
                {
                    numberOfTrainData = 2000;
                }
                count++;

                Classifier classifier = new Classifier();
                List<Sonuc> sonuc = new List<Sonuc>();
                DataTable dt = tableAbalone.Clone();
                tableAbalone = getTrainData(numberOfTrainData, dt, featureNumber);
                classifier.TrainClassifier(tableAbalone);
                string fileName = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), @"data\abalone_dataset.txt");
                const Int32 BufferSize = 128;
                using (var fileStream = File.OpenRead(fileName))
                using (var streamReader = new StreamReader(fileStream, Encoding.UTF8, true, BufferSize))
                {
                    String line;
                    int lineNumber = 0;
                    while ((line = streamReader.ReadLine()) != null)
                    {
                        lineNumber++;
                        if (numberOfTrainData >= lineNumber)
                        {
                            continue;
                        }
                        else
                        {
                            string[] splitLine = line.Split('\t');
                            double[] testData = new double[8];
                            Sonuc testSonuc = new Sonuc();
                            if (Convert.ToInt32(splitLine[splitLine.Length - 1]) == 1)
                            {
                                testSonuc.actual = "Young";
                            }
                            else if (Convert.ToInt32(splitLine[splitLine.Length - 1]) == 2)
                            {
                                testSonuc.actual = "Middle-Aged";
                            }
                            else
                            {
                                testSonuc.actual = "Old";

                            }
                            for (int i = 0; i < splitLine.Length - 1; i++)
                            {
                                if (splitLine[i].ToString() == "M")
                                {
                                    testData[i] = 1;
                                }
                                else if (splitLine[i].ToString() == "F")
                                {
                                    testData[i] = 2;
                                }
                                else if (splitLine[i].ToString() == "I")
                                {
                                    testData[i] = 3;
                                }
                                else
                                {
                                    testData[i] = double.Parse(splitLine[i], CultureInfo.InvariantCulture);
                                }
                            }
                            testSonuc.test = classifier.Classify(testData).ToString();
                            sonuc.Add(testSonuc);
                        }
                    }

                }
                 int testsData = 4177 - numberOfTrainData;
                 Console.Write("Number of Train Data: " + numberOfTrainData + " ,Number of Test Data: " + testsData);
                int[,] confusionMatrix = getConfusionMatrix(sonuc);
                WriteToConsoleConfusionMatrix(confusionMatrix);
                double errorRate= calculateErrorRate(confusionMatrix, sonuc.Count);
                Console.Write("\n Accuracy Rate:" + errorRate.ToString()+"\n");
                Console.Write("\n");


            }
            Console.ReadLine();
        }

        static DataTable getTrainData(int division, DataTable tableAbalone, int featureNumber)
        {
            string fileName = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), @"data\abalone_dataset.txt");

            const Int32 BufferSize = 128;
            int count = 0;
            using (var fileStream = File.OpenRead(fileName))
            using (var streamReader = new StreamReader(fileStream, Encoding.UTF8, true, BufferSize))
            {
                String line;
                while ((line = streamReader.ReadLine()) != null && count < division)
                {
                    count++;
                    DataRow dr = tableAbalone.NewRow();
                    string[] splitLine = line.Split('\t');
                    if (splitLine[0].ToString() == "M")
                    {
                        dr["Sex"] = 1;
                    }
                    else if (splitLine[0].ToString() == "F")
                    {
                        dr["Sex"] = 2;
                    }
                    else
                    {
                        dr["Sex"] = 3;
                    }
                    dr["Length"] = double.Parse(splitLine[1], CultureInfo.InvariantCulture);
                    dr["Diameter"] = double.Parse(splitLine[2], CultureInfo.InvariantCulture);
                    if (featureNumber == 2)
                    {
                        dr["Height"] = double.Parse(splitLine[3], CultureInfo.InvariantCulture);
                        dr["WholeWeight"] = double.Parse(splitLine[4], CultureInfo.InvariantCulture);
                        dr["Shucked"] = double.Parse(splitLine[5], CultureInfo.InvariantCulture);
                        dr["Viscera"] = double.Parse(splitLine[6], CultureInfo.InvariantCulture);
                        dr["Shellweight"] = double.Parse(splitLine[7], CultureInfo.InvariantCulture);
                    }
                    if (Convert.ToInt32(splitLine[8]) == 1)
                    {
                        dr["Class"] = "Young";
                    }
                    else if (Convert.ToInt32(splitLine[8]) == 2)
                    {
                        dr["Class"] = "Middle-Aged";
                    }
                    else
                    {
                        dr["Class"] = "Old";

                    }
                    tableAbalone.Rows.Add(dr);

                }
            }
            tableAbalone.TableName = division.ToString();
            return tableAbalone;
        }

        static int[,] getConfusionMatrix(List<Sonuc> sonuc)
        {
            int[,] x = new int[3, 3];
            foreach (var item in sonuc)
            {
                if (item.test == item.actual)
                {
                    if (item.actual == "Young")
                    {
                        x[0, 0] = x[0, 0] + 1;
                    }
                    if (item.actual == "Middle-Aged")
                    {
                        x[1, 1] = x[1, 1] + 1;
                    }
                    if (item.actual == "Old")
                    {
                        x[2, 2] = x[2, 2] + 1;
                    }

                }
                else
                {
                    if (item.actual == "Young" && item.test == "Middle-Aged")
                        x[0, 1] = x[0, 1] + 1;
                    if (item.actual == "Young" && item.test == "Old")
                        x[0, 2] = x[0, 2] + 1;
                    if (item.actual == "Middle-Aged" && item.test == "Young")
                        x[1, 0] = x[1, 0] + 1;
                    if (item.actual == "Middle-Aged" && item.test == "Old")
                        x[1, 2] = x[1, 2] + 1;
                    if (item.actual == "Old" && item.test == "Young")
                        x[2, 0] = x[2, 0] + 1;
                    if (item.actual == "Old" && item.test == "Middle-Aged")
                        x[2, 1] = x[2, 1] + 1;

                }
            }
            return x;
        }

        static void WriteToConsoleConfusionMatrix(int[,] matrix)
        {

            int rowLength = matrix.GetLength(0);
            int colLength = matrix.GetLength(1);

            Console.Write("\nConfusion Matrix\n");
            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    Console.Write(string.Format("{0} ", matrix[i, j]));
                }
                Console.Write(Environment.NewLine + Environment.NewLine);
            }
        }
        static double calculateErrorRate(int[,] matrix, int count)
        {

            int rowLength = matrix.GetLength(0);
            int colLength = matrix.GetLength(1);
            int wrongDataCount = 0;
            for (int i = 0; i < rowLength; i++)
            {
                for (int j = 0; j < colLength; j++)
                {
                    if (i == j) 
                    {
                        wrongDataCount += matrix[i, j];
                    }
                }
            }

            return (double)wrongDataCount/(double)count ;
        }
    }

    public class Sonuc
    {
        public string test { get; set; }
        public string actual { get; set; }
    }
}