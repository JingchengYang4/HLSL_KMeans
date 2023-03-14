using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;
using Newtonsoft.Json;
using Unity.Collections;
using UnityEngine.Experimental.Rendering;
using UnityEngine.UI;
using Random = UnityEngine.Random;
using System.Linq;
using Unity.Jobs;
using XCharts.Runtime;
using Debug = UnityEngine.Debug;

public class Main : MonoBehaviour
{
    public TextAsset mapJson;
    public TextAsset buildingJson;

    public MeshRenderer display;
    
    private RenderTexture mapRender;

    private NativeArray<Grid> grids = new NativeArray<Grid>();

    private ComputeBuffer gridBuffer;
    private ComputeBuffer scoreBuffer;

    public Text buildingLegendLabel;

    private int width;
    private int height;

    public ComputeShader batchClusteringShader;

    private int buildingCount = 4;

    private int[] buildingClusters;

    private string[] labels = new string[]
    {
        "住房", "农场", "滑雪场", "太阳能电池阵列", "户外运动场", "牧场", "农业旅游中心", "商场", "健身房", "美食节/饭店", "酒店", "学校(小学-高中)", "KTV",
        "书吧/书城/自习室 ", "公园 ", "大型游乐场"
    };

    public bool selectBuildings = false;
    private int[] selectedBuildings = new[] { 0, 1, 3, 5, 6 };

    private int[] coverTypeArea;

    public bool fixedSeed = false;
    public int currentSeed;

    public Text seedLabel;

    void Start()
    {
        if (!fixedSeed)
        {
            currentSeed = Random.Range(0, int.MaxValue);
        }
        seedLabel.text = currentSeed.ToString();
        Random.InitState(currentSeed);

        InitializeForKMeans();

        buildingClusters = new int[buildingCount];
        for (int i = 0; i < buildingCount; i++)
        {
            buildingClusters[i] = 0;
            buildingLegendLabel.text += $"{labels[selectedBuildings[i]]} ";
        }

        if (geneticAlgorithm)
        {
            InitializeForGeneticAlgorithm();
        }

        if (sensitivityAnalysis)
        {
            InitializeForSensitivityAnalysis();
        }

        float maxSize = Mathf.Max(width, height)/100f;


        display.transform.localScale = new Vector3((width / 100f)/maxSize*0.93f, 1, (height / 100f)/maxSize * 0.93f);
    }

    private void Update()
    {
        if (geneticAlgorithm)
        {
            
            if (recordEvolution && Input.GetKeyDown(KeyCode.S))
            {
                TakeScreenshot();
            }
            
            timer -= Time.deltaTime;
            if (!evolving && (timer <= 0 || evolutionRate <= 0))
            {
                timer = evolutionRate;
                evolving = true;
                if (evolutionIteration >= 500 && sensitivityAnalysis)
                {
                    ConductSensitivityAnalysis();
                    return;
                }
                StartCoroutine(Evolution());
            }
            
            return;
        }

        if (distributionAnalysis && Input.GetKeyDown(KeyCode.D))
        {
            //CalculateDistributions();
            var data = totalCosts.OrderBy(x => x).ToList();
            VisualizeDistribution(data);
            Debug.Log(BoxAndWhisker(data));
        }
    }

    #region GeneticAlgorithm

    [Header("Genetic Algorithm")]
    public bool geneticAlgorithm = false;
    
    public float evolutionRate = 1f;
    public float timer = 5;
    public int maxCentroidMutateOffset = 8;

    private bool evolving = false;
    private int evolutionIteration = 0;
    private int evolutionIterationSinceLastChange = 0;
    private float bestIndex = 0;
    
    public int populationSize = 100;
    private List<Gene> population = new List<Gene>();

    public Text evolutionLabel;
    
    private string screenshotDirectory = "/Users/jingchengyang/Documents/Land Cover Data/";

    public bool recordEvolution = false;

    public void InitializeForGeneticAlgorithm()
    {
        if (recordEvolution && !Directory.Exists(screenshotDirectory + currentSeed))
        {
            Directory.CreateDirectory(screenshotDirectory + currentSeed);
        }
        
        for (int i = 0; i < populationSize; i++)
        {
            var gene = new Gene();
            gene.centroids = GenerateRandomCentroid();
            population.Add(gene);
        }
    }
    
    IEnumerator Evolution()
    {
        evolutionIteration++;

        ClusterKMeansBatch();

        population = population.OrderByDescending(x => x.index).ToList();

        yield return null;

        evolutionIterationSinceLastChange++;

        if (sensitivityAnalysis)
        {
            if (evolutionIteration % sensitivitySamplingInterval == 0)
            {
                saScore.Add(population[0].index);
                landPercent.Add(population[0].landUsePercentage);
            }
        }

        if (population[0].index > bestIndex)
        {
            bestIndex = population[0].index;
            //Debug.Log(bestIndex);
            evolutionIterationSinceLastChange = 0;
            if (recordEvolution)
            {
                TakeScreenshot();
            }
        }

        for (int i = populationSize / 2; i < populationSize; i++)
        {
            int father = Random.Range(0, populationSize / 2);
            int mother = Random.Range(0, populationSize / 2);
            while (father == mother)
            {
                mother = Random.Range(0, populationSize / 2);
            }
                    
            population[i].centroids.Dispose();

            population[i].centroids = CrossOverCentroids(population[father].centroids, population[mother].centroids, Random.Range(1, kCount - 1));
                    
            MutateCentroid(population[i].centroids);
        }

        evolutionLabel.text = $"Evolution Iteration: {evolutionIteration}\nHighest Index: {bestIndex}\nIterations since last change: {evolutionIterationSinceLastChange}";

        evolving = false;
    }

    public void TakeScreenshot()
    {
        if (sensitivityAnalysis)
        {
            ScreenCapture.CaptureScreenshot($"{screenshotDirectory}{currentSeed}/S{(saProgress*100).ToString("F2")}%_ITER{evolutionIteration}.png");
        }
        else
        {
            ScreenCapture.CaptureScreenshot($"{screenshotDirectory}{currentSeed}/{evolutionIteration}.png");
        }
    }

    NativeArray<Center> GenerateRandomCentroid()
    {
        HashSet<int> kIndex = new HashSet<int>();

        var centeroids = new NativeArray<Center>(kCount, Allocator.Persistent);

        for (int i = 0; i < kCount; i++)
        {
            int gridIndex = GetIndex(Random.Range(0, width), Random.Range(0, height));
            var grid = grids[gridIndex];
            while (grid.cover <= 0 || kIndex.Contains(gridIndex))
            {
                gridIndex = GetIndex(Random.Range(0, width), Random.Range(0, height));
                grid = grids[gridIndex];
            }

            kIndex.Add(gridIndex);
            centeroids[i] = new Center(grid.x, grid.y, Random.Range(0, buildingCount));
        }

        return centeroids;
    }

    NativeArray<Center> CrossOverCentroids(NativeArray<Center> a, NativeArray<Center> b, int point)
    {
        var centroids = new NativeArray<Center>(kCount, Allocator.Persistent);
        
        for (int i = 0; i < kCount; i++)
        {
            if (i < point)
            {
                centroids[i] = a[i];
            }
            else
            {
                centroids[i] = b[i];
            }
        }

        return centroids;
    }

    public float geneMutationChance = 0.05f;

    public void MutateCentroid(NativeArray<Center> centroids)
    {
        for (int i = 0; i < kCount; i++)
        {
            if (Random.value < geneMutationChance)
            {
                Center newCenter = new Center(centroids[i].x + Random.Range(-maxCentroidMutateOffset, maxCentroidMutateOffset+1), centroids[i].y + Random.Range(-maxCentroidMutateOffset, maxCentroidMutateOffset+1), Random.Range(0, buildingCount));
                while (newCenter.y * width + newCenter.x < 0 || newCenter.y * width + newCenter.x >= grids.Length || grids[newCenter.y * width + newCenter.x].cover <= 0)
                {
                    newCenter = new Center(centroids[i].x + Random.Range(-maxCentroidMutateOffset, maxCentroidMutateOffset), centroids[i].y + Random.Range(-maxCentroidMutateOffset, maxCentroidMutateOffset), Random.Range(0, buildingCount));
                }

                centroids[i] = newCenter;
            }
        }
    }

    public void ResetEvolution()
    {
        StopAllCoroutines();
        Random.InitState(currentSeed);
        for (int i = 0; i < populationSize; i++)
        {
            population[i].centroids.Dispose();
            population[i].centroids = GenerateRandomCentroid();
        }
        evolutionIteration = 0;
        bestIndex = 0;
        evolutionIterationSinceLastChange = 0;
        evolving = false;
    }
    
    #endregion

    #region SensitivityAnalysis

    [Header("Sensitivity Analysis")]
    public bool sensitivityAnalysis;
    
    public SensitivityAnalysis saType;

    private List<string> landPercents;
    private List<float> landPercent;

    private List<string> saScores;
    private List<float> saScore;

    public float environmentHarmIndexModifier = 0;
    public float longShortTermModifier = 0;
    public int kCountModifier = 0;

    public int sensitivitySamplingInterval = 5;

    private float saProgress;

    public int question = 1;

    public void InitializeForSensitivityAnalysis()
    {
        landPercents = new List<string>();
        landPercent = new List<float>();

        saScores = new List<string>();
        saScore = new List<float>();

        if (saType == SensitivityAnalysis.LongShortTerm)
        {
            longShortTermModifier = -0.1f;
        }
        else if (saType == SensitivityAnalysis.DistancePower)
        {
            distancePower = 0.05f;
        }
    }

    public void ConductSensitivityAnalysis()
    {
        saScores.Add(string.Join(", ", saScore));
        saScore.Clear();
        
        switch (saType)
        {
            case SensitivityAnalysis.Environment:
                saProgress = environmentHarmIndexModifier / 0.02f;
         
                landPercents.Add(string.Join(", ", landPercent));
                landPercent.Clear();
                if (environmentHarmIndexModifier < 0.02f)
                {
                    ResetEvolution();
                }
                else
                {
                    File.WriteAllLines(screenshotDirectory + $"{currentSeed} environment sensitivity analysis (index).txt", saScores);
                    File.WriteAllLines(screenshotDirectory + $"{currentSeed} environment sensitivity analysis (land usage).txt", landPercents);
                }
                environmentHarmIndexModifier += 0.001f;
                
                break;
            
            case SensitivityAnalysis.LongShortTerm:
                saProgress = (longShortTermModifier+0.1f) / 0.2f;
                if (longShortTermModifier < 0.1f)
                {
                    ResetEvolution();
                }
                else
                {
                    File.WriteAllLines(screenshotDirectory + $"{currentSeed} long-short-term sensitivity analysis (index).txt", saScores);
                }
                longShortTermModifier += 0.02f;
                break;
            
            case SensitivityAnalysis.K:
                saProgress = kCountModifier/15f;
                if(kCountModifier < 15f) ResetEvolution();
                else
                {
                    File.WriteAllLines(screenshotDirectory + $"{currentSeed} k sensitivity analysis (index).txt", saScores);
                }
                kCountModifier += 1;
                break;
            case SensitivityAnalysis.DistancePower:
                saProgress = (distancePower - 0.05f) / 20;
                if(distancePower < 1) ResetEvolution();
                else
                {
                    File.WriteAllLines(screenshotDirectory + $"{currentSeed} distance power sensitivity analysis (index).txt", saScores);
                }

                distancePower += 0.05f;
                break;
        }
        
        Debug.Log($"Sensitivity Analysis {saProgress*100}%");
    }
    
    #endregion

    #region BatchKMeans

    [Header("Batch K-Means")]
    public int kCount = 5;
    public int totalIterations = 10000;
    public float distancePower = 0.1f;
    
    public bool removeLowClusters = false;
    public int clusterRemoveThreshold = 4;
    
    public Text annotation;

    ComputeBuffer landUsageBatchBuffer; 
    ComputeBuffer centroidsBatchBuffer;
    ComputeBuffer kCountBuffer;
    ComputeBuffer buildingAreaBatchBuffer;

    private int batchKernel = 0;

    private NativeArray<int> nativeCoverTypeArea;

    private NativeArray<float> indices;
    private NativeArray<float> landUsePercentage;

    public void InitializeForKMeans()
    {
        float[,] map = JsonConvert.DeserializeObject<float[,]>(mapJson.text);
        
        width = map.GetLength(1); 
        height = map.GetLength(0);

        mapRender = new RenderTexture(width, height, 0, GraphicsFormat.R32_SFloat);
        mapRender.enableRandomWrite = true;
        mapRender.filterMode = FilterMode.Point;
        mapRender.Create();

        RenderTexture.active = mapRender;

        grids = new NativeArray<Grid>(width * height, Allocator.Persistent);

        coverTypeArea = Enumerable.Repeat(0, 4).ToArray();
        
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                int coverType = (int) map[height - 1 - y, x];
                if(coverType > 0) coverTypeArea[coverType - 1]++;
                grids[y * width + x] = new Grid(x, y, coverType);
            }
        }

        gridBuffer = new ComputeBuffer(grids.Length, Marshal.SizeOf(typeof(Grid)));
        
        float[,] scores = JsonConvert.DeserializeObject<float[,]>(buildingJson.text);
        
        buildingCount = scores.GetLength(1);

        if (selectBuildings)
        {
            buildingCount = selectedBuildings.Length;
        }
        else
        {
            selectedBuildings = Enumerable.Range(0, buildingCount).ToArray();
        }

        int scoreSize = buildingCount * 4;

        scoreBuffer = new ComputeBuffer(scoreSize, sizeof(float));

        NativeArray<float> nativeScores = new NativeArray<float>(scoreSize, Allocator.Temp);
        for (int i = 0; i < buildingCount; i++)
        {
            string line = "";
            for (int j = 0; j < 4; j++)
            {
                nativeScores[i * 4 + j] = scores[j, selectedBuildings[i]];
                line += $"{nativeScores[i * 4 + j]} ";
            }
            //Debug.Log(line);
        }

        scoreBuffer.SetData(nativeScores);

        nativeScores.Dispose();

        gridBuffer.SetData(grids);

        display.material.SetTexture("_Map", mapRender);
        
        landUsageBatchBuffer = new ComputeBuffer(populationSize * 4, sizeof(int));
        buildingAreaBatchBuffer = new ComputeBuffer(populationSize * buildingCount, sizeof(int));
        
        kCountBuffer  = new ComputeBuffer(populationSize, sizeof(int), ComputeBufferType.Structured, ComputeBufferMode.SubUpdates);
        centroidsBatchBuffer = new ComputeBuffer(populationSize * kCount, Marshal.SizeOf(typeof(Center)), ComputeBufferType.Structured, ComputeBufferMode.SubUpdates);

        batchKernel = batchClusteringShader.FindKernel("BatchClustering");

        batchClusteringShader.SetInt("width", width);
        batchClusteringShader.SetInt("height", height);
        batchClusteringShader.SetInt("layer_size", populationSize);
        batchClusteringShader.SetBuffer(batchKernel, "GridBuffer", gridBuffer);
        batchClusteringShader.SetBuffer(batchKernel, "ScoreBuffer", scoreBuffer);
        batchClusteringShader.SetBuffer(batchKernel, "LandUsageBatchBuffer", landUsageBatchBuffer);
        batchClusteringShader.SetBuffer(batchKernel, "KCountBuffer", kCountBuffer);
        batchClusteringShader.SetBuffer(batchKernel, "CentroidsBatchBuffer", centroidsBatchBuffer);
        batchClusteringShader.SetBuffer(batchKernel, "BuildingAreaBuffer", buildingAreaBatchBuffer);
        batchClusteringShader.SetTexture(batchKernel, "display", mapRender);

        nativeCoverTypeArea = new NativeArray<int>(coverTypeArea, Allocator.Persistent);
        
        saScores = new List<string>();
        saScore = new List<float>();
        landPercent = new List<float>();
        landPercents = new List<string>();
        
        indices = new NativeArray<float>(populationSize, Allocator.Persistent);
        landUsePercentage = new NativeArray<float>(populationSize, Allocator.Persistent);
    }

    public void ClusterKMeansBatch()
    {
        batchClusteringShader.SetFloat("distance_power", distancePower);
        batchClusteringShader.SetInt("building_count", buildingCount);
        batchClusteringShader.SetInt("global_k_count", kCount);
        
        annotation.text = $"Batch Cluster: {populationSize}\nK = {kCount-kCountModifier}\nDistance Power = {distancePower}";
        
        kCountBuffer.SetData(Enumerable.Repeat(kCount-kCountModifier, populationSize).ToArray());

        for (int i = 0; i < populationSize; i++)
        {
            centroidsBatchBuffer.SetData(population[i].centroids, 0, i * kCount, kCount);
        }

        Center[] centroidsRead;
        
        for (int iter = 0; iter < totalIterations; iter++)
        {
            buildingAreaBatchBuffer.SetData(Enumerable.Repeat(0, buildingCount * populationSize).ToArray());
            landUsageBatchBuffer.SetData(Enumerable.Repeat(0, 4 * populationSize).ToArray());
            
            batchClusteringShader.Dispatch(batchKernel, Mathf.FloorToInt(width/8f), Mathf.FloorToInt(height/8f), 13);
            //batchClusteringShader.Dispatch(batchKernel, Mathf.FloorToInt(grids.Length/8f), 13, 1);

            centroidsRead = new Center[kCount * populationSize];
            centroidsBatchBuffer.GetData(centroidsRead);

            var centroidsWriter = centroidsBatchBuffer.BeginWrite<Center>(0, kCount * populationSize);
            var kCountWriter = kCountBuffer.BeginWrite<int>(0, populationSize);

            for (int j = 0; j < populationSize; j++)
            {
                int validCentroids = 0;

                for (int i = 0; i < kCount-kCountModifier; i++)
                {
                    int index = j * kCount + i;
                    
                    if (centroidsRead[index].totalX <= clusterRemoveThreshold && removeLowClusters)
                    {
                        continue;
                    }

                    if (centroidsRead[index].totalX <= 0)
                    {
                        validCentroids++;
                        continue;
                    }

                    Center newCenter = new Center(Mathf.RoundToInt(centroidsRead[index].cumX * 1.0f / centroidsRead[index].totalX),
                        Mathf.RoundToInt(centroidsRead[index].cumY * 1.0f / centroidsRead[index].totalY), centroidsRead[index].buildingType);
                    
                    centroidsWriter[j * kCount + validCentroids++] = newCenter;
                }

                kCountWriter[j] = validCentroids;
            }
            
            kCountBuffer.EndWrite<int>(populationSize);
            centroidsBatchBuffer.EndWrite<Center>(kCount * populationSize);
        }
        
        int[] landUsageRead = new int[4 * populationSize];
        landUsageBatchBuffer.GetData(landUsageRead);

        int[] buildingAreaRead = new int[buildingCount * populationSize];
        buildingAreaBatchBuffer.GetData(buildingAreaRead);
        
        var ahpJob = new AhpIndexJob();
        
        var nativeBuildingAreas = new NativeArray<int>(buildingAreaRead, Allocator.Persistent);
        var nativeLandUsage = new NativeArray<int>(landUsageRead, Allocator.Persistent);

        ahpJob.indices = indices;
        ahpJob.coverTypeArea = nativeCoverTypeArea;
        ahpJob.buildingAreas = nativeBuildingAreas;
        ahpJob.landUsage = nativeLandUsage;
        ahpJob.buildingCount = buildingCount;
        ahpJob.environmentHarmIndexModifier = environmentHarmIndexModifier;
        ahpJob.landUsePercentage = landUsePercentage;
        ahpJob.longShortTermModifier = longShortTermModifier;
        ahpJob.question = question;
            
        var handler = ahpJob.Schedule(populationSize, 16);
        handler.Complete();

        for (int i = 0; i < populationSize; i++)
        {
            //float legacyAhp = GetAhpIndex(landUsageRead, buildingAreaRead, i);
            //Debug.Log($"{legacyAhp - indices[i]}");
            population[i].index = indices[i];
            population[i].landUsePercentage = landUsePercentage[i];
            //population[i].index = legacyAhp;
        }
        
        nativeBuildingAreas.Dispose();
        nativeLandUsage.Dispose();
    }

    public void DisposeForBatching()
    {
        landUsageBatchBuffer.Dispose();;
        centroidsBatchBuffer.Dispose();
        kCountBuffer.Dispose();
        buildingAreaBatchBuffer.Dispose();
        nativeCoverTypeArea.Dispose();
        indices.Dispose();
        landUsePercentage.Dispose();
    }
    
    #endregion

    #region DistributionAnalysis
    
    [Header("Distribution Analysis")]
    public bool distributionAnalysis = false;
    
    private List<float> totalProfits = new List<float>();
    private List<float> totalCosts = new List<float>();
    private List<float> totalCommunityNeeds = new List<float>();
    
    
    public LineChart lineChart;
    public int plotIntervals = 400;

    void VisualizeDistribution(List<float> values)
    {
        int interval = (int)(values.Last()/plotIntervals);

        int[] intervals = Enumerable.Repeat(0, plotIntervals+1).ToArray();
        foreach (var value in values)
        {
            intervals[(int) value / interval]++;
        }
        
        lineChart.ClearData();
        for (int i = 0; i < plotIntervals; i++)
        {
            if(intervals[i] <= 0) continue;
            lineChart.AddXAxisData((i*interval).ToString());
            lineChart.AddData(0, intervals[i]);
        }
    }

    string BoxAndWhisker(List<float> values)
    {
        //values should be sorted
        float min = values.First();
        float max = values.Last();
        float lowerQ = values[(int) (values.Count * 0.25f)];
        float upperQ = values[(int) (values.Count * 0.75f)];
        float median = values[(int) (values.Count * 0.5f)];
        
        return $"{min} | {lowerQ} {median} {upperQ} | {max}";
    }
    
    #endregion

    public int GetIndex(int x, int y)
    {
        return y * width + x;
    }

    private void OnApplicationQuit()
    {
        grids.Dispose();
        gridBuffer.Dispose();
        scoreBuffer.Dispose();
        //if(centeroidsBuffer != null && centeroidsBuffer.IsValid()) centeroidsBuffer.Dispose();

        string oldDir = $"{screenshotDirectory}{currentSeed}";
        
        string newDir = $"{screenshotDirectory}ITER{evolutionIteration}_K{kCount}_D{distancePower}_SEED{currentSeed}";

        if (sensitivityAnalysis)
        {
            newDir = $"{screenshotDirectory}SA_{Enum.GetName(typeof(SensitivityAnalysis), saType)}_{currentSeed}_{saProgress*100}%";
        }
        
        Directory.Move(oldDir, newDir);

        foreach (var gene in population)
        {
            gene.centroids.Dispose();
        }

        DisposeForBatching();
    }
}
