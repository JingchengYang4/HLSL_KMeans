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
    private ComputeBuffer landUsageBuffer;

    public Text buildingLegendLabel;

    private int kernel;

    private int width;
    private int height;

    public ComputeShader batchClusteringShader;
    public ComputeShader clusteringShader;

    private int buildingCount = 4;

    private int[] buildingClusters;

    private string[] labels = new string[]
    {
        "住房", "农场", "滑雪场", "太阳能电池阵列", "户外运动场", "牧场", "农业旅游中心", "商场", "健身房", "美食节/饭店", "酒店", "学校(小学-高中)", "KTV",
        "书吧/书城/自习室 ", "公园 ", "大型游乐场"
    };

    public bool selectBuildings = false;
    private int[] selectedBuildings = new[] { 0, 1, 3, 5, 6 };

    private bool inProcess = false;

    public Text annotation;

    public int totalIterations = 10000;
    public int kCount = 5;
    public float distancePower = 0.1f;
    
    public bool calculateMostClustered = false;

    public int clusterRemoveThreshold = 4;
    public bool removeLowClusters = false;

    public float iterationUpdateRate = 0.01f;

    public bool continuousCluster = false;

    public bool evaluateMetrics = false;

    private List<float> totalProfits = new List<float>();
    private List<float> totalCosts = new List<float>();
    private List<float> totalCommunityNeeds = new List<float>();

    public bool distributionAnalysis = false;

    public int numClusters = 0;

    public LineChart lineChart;

    private int[] coverTypeArea;

    public int populationSize = 100;
    private List<Gene> population = new List<Gene>();

    public int currentSeed;

    public Text seedLabel;
    
    private string screenshotDirectory = "/Users/jingchengyang/Documents/Land Cover Data/";

    public bool recordEvolution = false;

    void Start()
    {
        currentSeed = Random.Range(0, int.MaxValue);
        seedLabel.text = currentSeed.ToString();
        Random.InitState(currentSeed);

        if (recordEvolution && !Directory.Exists(screenshotDirectory + currentSeed))
        {
            Directory.CreateDirectory(screenshotDirectory + currentSeed);
        }
        
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
        landUsageBuffer = new ComputeBuffer(4, sizeof(int));

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

        kernel = clusteringShader.FindKernel("Clustering");
        
        gridBuffer.SetData(grids);
        
        clusteringShader.SetBuffer(kernel, "GridBuffer", gridBuffer);
        clusteringShader.SetBuffer(kernel, "ScoreBuffer", scoreBuffer);
        clusteringShader.SetBuffer(kernel, "LandUsageBuffer", landUsageBuffer);
        clusteringShader.SetTexture(kernel, "display", mapRender);
        clusteringShader.SetInt("width", width);
        clusteringShader.SetInt("height", height);
        
        display.material.SetTexture("_Map", mapRender);

        buildingClusters = new int[buildingCount];
        for (int i = 0; i < buildingCount; i++)
        {
            buildingClusters[i] = 0;
            buildingLegendLabel.text += $"{labels[selectedBuildings[i]]} ";
        }

        if (geneticAlgorithm)
        {
            for (int i = 0; i < populationSize; i++)
            {
                var gene = new Gene();
                gene.centroids = GenerateRandomCentroid();
                population.Add(gene);
            }
        }

        InitializeForBatching();
    }

    public Vector3 profitQuartiles;
    public Vector3 costQuartiles;
    public Vector3 communityNeedsQuartiles;

    private float clusterIndex = 0;
    
    //GenerateRandomCentroid

    public float evolutionRate = 1f;
    public float timer = 5;
    public bool geneticAlgorithm = false;
    public int maxCentroidMutateOffset = 8;

    private bool evolving = false;
    private int evolutionIteration = 0;
    private int evolutionIterationSinceLastChange = 0;
    private float bestIndex = 0;

    public Text evolutionLabel;

    private void Update()
    {
        if (geneticAlgorithm)
        {
            timer -= Time.deltaTime;
            if (!evolving && (timer <= 0 || evolutionRate <= 0))
            {
                timer = evolutionRate;
                evolving = true;
                StartCoroutine(Evolution());
            }
            
            return;
        }
        
        if ((Input.GetKeyDown(KeyCode.C) || continuousCluster) && !inProcess)
        {
            //inProcess = true;
            //StartCoroutine(ClusterKMeans());
        }

        if (distributionAnalysis && Input.GetKeyDown(KeyCode.D))
        {
            //CalculateDistributions();
            var data = totalCosts.OrderBy(x => x).ToList();
            VisualizeDistribution(data);
            Debug.Log(BoxAndWhisker(data));
        }
    }

    public bool BatchProcess = true;

    IEnumerator Evolution()
    {
        evolutionIteration++;
        
        ClusterKMeansBatch();

        population = population.OrderByDescending(x => x.index).ToList();

        yield return null;

        evolutionIterationSinceLastChange++;

        if (population[0].index > bestIndex)
        {
            bestIndex = population[0].index;
            Debug.Log(bestIndex);
            evolutionIterationSinceLastChange = 0;
            if (recordEvolution)
            {
                yield return null;
                ScreenCapture.CaptureScreenshot($"{screenshotDirectory}{currentSeed}/{evolutionIteration}.png");
            }
            
            yield return null;
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
                    newCenter = new Center(centroids[i].x + Random.Range(-3, 4), centroids[i].y + Random.Range(-3, 4), Random.Range(0, buildingCount));
                }

                centroids[i] = newCenter;
            }
        }
    }

    ComputeBuffer landUsageBatchBuffer; 
    ComputeBuffer centroidsBatchBuffer;
    ComputeBuffer kCountBuffer;
    ComputeBuffer buildingAreaBatchBuffer;

    private int batchKernel = 0;

    private NativeArray<int> nativeCoverTypeArea;

    public void InitializeForBatching()
    {
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
    }

    public void ClusterKMeansBatch()
    {
        batchClusteringShader.SetFloat("distance_power", distancePower);
        batchClusteringShader.SetInt("building_count", buildingCount);
        batchClusteringShader.SetInt("global_k_count", kCount);
        
        annotation.text = $"Batch Cluster: {populationSize}\nK = {kCount}\nDistance Power = {distancePower}";
        
        kCountBuffer.SetData(Enumerable.Repeat(kCount, populationSize).ToArray());

        for (int i = 0; i < populationSize; i++)
        {
            centroidsBatchBuffer.SetData(population[i].centroids, 0, i * kCount, kCount);
        }

        Center[] centroidsRead;
        
        for (int iter = 0; iter < totalIterations; iter++)
        {
            buildingAreaBatchBuffer.SetData(Enumerable.Repeat(0, buildingCount * populationSize).ToArray());
            landUsageBatchBuffer.SetData(Enumerable.Repeat(0, 4 * populationSize).ToArray());
            
            batchClusteringShader.Dispatch(batchKernel, 12, 11, 13);
            //batchClusteringShader.Dispatch(batchKernel, Mathf.FloorToInt(grids.Length/8f), 13, 1);

            centroidsRead = new Center[kCount * populationSize];
            centroidsBatchBuffer.GetData(centroidsRead);

            var centroidsWriter = centroidsBatchBuffer.BeginWrite<Center>(0, kCount * populationSize);
            var kCountWriter = kCountBuffer.BeginWrite<int>(0, populationSize);

            for (int j = 0; j < populationSize; j++)
            {
                int validCentroids = 0;

                for (int i = 0; i < kCount; i++)
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
        NativeArray<float> indices = new NativeArray<float>(populationSize, Allocator.Persistent);
        var nativeBuildingAreas = new NativeArray<int>(buildingAreaRead, Allocator.Persistent);
        var nativeLandUsage = new NativeArray<int>(landUsageRead, Allocator.Persistent);
            
        ahpJob.indices = indices;
        ahpJob.coverTypeArea = nativeCoverTypeArea;
        ahpJob.buildingAreas = nativeBuildingAreas;
        ahpJob.landUsage = nativeLandUsage;
        ahpJob.buildingCount = buildingCount;
            
        var handler = ahpJob.Schedule(populationSize, 16);
        handler.Complete();

        for (int i = 0; i < populationSize; i++)
        {
            //float legacyAhp = GetAhpIndex(landUsageRead, buildingAreaRead, i);
            //Debug.Log($"{legacyAhp - indices[i]}");
            population[i].index = indices[i];
            //population[i].index = legacyAhp;
        }

        indices.Dispose();
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
    }

    public IEnumerator ClusterKMeans(NativeArray<Center> centroidsRef)
    {
        if(!geneticAlgorithm) numClusters++;
        clusteringShader.SetInt("k_count", kCount);
        clusteringShader.SetFloat("distance_power", distancePower);
        clusteringShader.SetInt("building_count", buildingCount);

        for (int i = 0; i < buildingCount; i++)
        {
            buildingClusters[i] = 0;
        }

        NativeArray<Center> centroids = new NativeArray<Center>(centroidsRef, Allocator.Persistent);

        ComputeBuffer centeroidsBuffer = new ComputeBuffer(kCount, Marshal.SizeOf(typeof(Center)));
        centeroidsBuffer.SetData(centroids);

        clusteringShader.SetBuffer(kernel, "CentroidsBuffer", centeroidsBuffer);

        Center[] centroidsRead = new Center[kCount];

        for (int iter = 0; iter < totalIterations; iter++)
        {
            landUsageBuffer.SetData(Enumerable.Repeat(0, 4).ToArray());
            
            clusteringShader.Dispatch(kernel, 12, 11, 1);

            centroidsRead = new Center[kCount];
            centeroidsBuffer.GetData(centroidsRead);

            int changes = 0;

            int validCentroids = 0;

            for (int i = 0; i < kCount; i++)
            {
                if (centroidsRead[i].totalX <= clusterRemoveThreshold && removeLowClusters)
                {
                    continue;
                }

                if (centroidsRead[i].totalX <= 0)
                {
                    validCentroids++;
                    continue;
                }

                Center newCenter = new Center(Mathf.RoundToInt(centroidsRead[i].cumX * 1.0f / centroidsRead[i].totalX),
                    Mathf.RoundToInt(centroidsRead[i].cumY * 1.0f / centroidsRead[i].totalY), centroidsRead[i].buildingType);
                if (newCenter.x != centroids[i].x || newCenter.y != centroids[i].y)
                {
                    changes++;
                }
                centroids[validCentroids++] = newCenter;
            }

            if (changes <= 0)
            {
                annotation.text = $"Converged at :{iter + 1}/{totalIterations}\nK = {kCount}\nDistance Power = {distancePower}";
                break;
            }
            
            centeroidsBuffer.SetData(centroids);
            
            clusteringShader.SetInt("k_count", validCentroids);

            annotation.text = $"Iterations:{iter + 1}/{totalIterations}\nK = {kCount}\nDistance Power = {distancePower}";

            if (iterationUpdateRate > 0)
            {
                yield return new WaitForSeconds(iterationUpdateRate);
            }
        }

        if (calculateMostClustered || evaluateMetrics)
        {
            for (int i = 0; i < kCount; i++)
            {
                buildingClusters[centroidsRead[i].buildingType] += centroidsRead[i].totalX;
            }
        }
        
        if(calculateMostClustered)
        {
            Debug.Log($"{string.Join(' ', buildingClusters)}");

            var sorted = buildingClusters
                .Select((x, i) => new KeyValuePair<int, int>(x, i))
                .OrderByDescending(x => x.Key)
                .ToList();

            string output = "Most Clustered Buildings: ";
            for (int j = 0; j < buildingCount; j++)
            {
                output += $"{labels[selectedBuildings[sorted.ElementAt(j).Value]]}, ";
            }

            Debug.Log(output);
        }

        if (evaluateMetrics)
        {
            int[] landUsage = new int[4];
            landUsageBuffer.GetData(landUsage);
            clusterIndex = GetAhpIndex(landUsage, buildingClusters);
        }

        centroids.Dispose();
        centeroidsBuffer.Dispose();

        inProcess = false;
    }

    public float GetAhpIndex(int[] landUsage, int[] buildingAreas, int startIndex = 0)
    {
        float profits = 0;
        float costs = 0;
        float paybackIndices = 0;
        float longTermDevelopments = 0;
        float developmentDurationIndices = 0;
        float communityNeeds = 0;

        int totalArea = 0;

        for (int i = 0; i < buildingCount; i++)
        {
            float profit = 0;
            float cost = 0;
            float paybackDuration = 0;
            float longTermDevelopment = 0;
            float constructionDuration = 0;
            float communityNeed = 0;
            if (i == 0)//housing
            {
                profit = 223.8f;
                //there are two values?
                cost = 2152.85f;
                paybackDuration = 10.6884f;
                longTermDevelopment = 1.2f;
                constructionDuration = 1;
                communityNeed = 1f;
            }
            else if (i == 1)//farms
            {
                
                profit = 0.3153f;
                cost = 0.7809f;
                paybackDuration = 2.4767f;
                longTermDevelopment = 1f;
                constructionDuration = 1;
                communityNeed = 0.6f;
            }
            else if (i == 2)//solar farms
            {
                profit = 13.9931f;
                cost = 46.3924f;
                paybackDuration = 3.3154f;
                longTermDevelopment = 1f;
                constructionDuration = 1f;
                communityNeed = 0f;
            }
            else if (i == 3)//ranch
            {
                profit = 0.2231f;
                cost = 0.7136f;
                paybackDuration = 3.1986f;
                longTermDevelopment = 1f;
                constructionDuration = 1 / 6f;
                communityNeed = 0.6f;
            }
            else if (i == 4)//agricultural visitor
            {
                profit = 2.4711f;
                cost = 12.3553f;
                paybackDuration = 5f;
                longTermDevelopment = 0.7f;
                constructionDuration = 3f;
                communityNeed = 1.7f;
            }

            int buildingArea = buildingAreas[startIndex * buildingCount +i];
            
            profit *= buildingArea;
            cost *= buildingArea;
            //communityNeed *= buildingArea;

            profits += profit;
            costs += cost;

            totalArea += buildingArea;
            
            paybackIndices += PaybackIndex(paybackDuration) * buildingArea;
            
            longTermDevelopments += longTermDevelopment/2 * buildingArea;
            
            developmentDurationIndices += DevelopmentDurationIndex(constructionDuration) * buildingArea;
            communityNeeds += communityNeed * buildingArea;
        }

        if (distributionAnalysis)
        {
            totalProfits.Add(profits);
            totalCosts.Add(costs);
            totalCommunityNeeds.Add(communityNeeds);
        }

        float environmentHarmIndex = 0;
        environmentHarmIndex += landUsage[startIndex * 4 + 0] * 1.0f / coverTypeArea[0] * 4;
        environmentHarmIndex += landUsage[startIndex * 4 + 2] * 1.0f / coverTypeArea[2] * 3;
        environmentHarmIndex += landUsage[startIndex * 4 + 1] * 1.0f / coverTypeArea[1] * 2;
        environmentHarmIndex /= 9f;

        /*for (int i = 0; i < 4; i++)
        {
            Debug.Log($"{landUsage[i]} / {coverTypeArea[i]}");
        }*/
        
        float profitIndex = EvaluateDistribution(profits, profitQuartiles);
        float constructionCostIndex = EvaluateDistribution(costs, costQuartiles);
        //Debug.Log(costs);
        float developmentDurationIndex = developmentDurationIndices / totalArea;
        float paybackIndex = paybackIndices / totalArea;
        float communityFitIndex = communityNeeds / (2f * totalArea);
        float longTermDevelopmentIndex = longTermDevelopments / totalArea;

        //Debug.Log($"Average Profit Index {profits.Average()}");
        
        //Debug.Log($"{profitIndex} {paybackIndex} {constructionCostIndex} {longTermDevelopmentIndex} {environmentHarmIndex} {communityFitIndex}");

        float index =
            0.379f * profitIndex +
            0.179f * paybackIndex +
            0.122f * constructionCostIndex +
            0.179f * longTermDevelopmentIndex +
            0.066f * developmentDurationIndex -
            0.032f * environmentHarmIndex +
            0.042f * communityFitIndex;

        return index;
    }

    float PaybackIndex(float paybackYear)
    {
        return 1 - 1 / (1 + Mathf.Exp(-(3 / 10 * paybackYear + 4.5f)));
    }

    float DevelopmentDurationIndex(float constructionDuration)
    {
        return Mathf.Exp(-0.6f * constructionDuration);
    }

    float EvaluateDistribution(float value, Vector3 distribution)
    {
        if (value < distribution.x)
        {
            return Mathf.InverseLerp(0, distribution.x, value);
        }
        else if (value < distribution.y)
        {
            return Mathf.InverseLerp(distribution.x, distribution.y, value);
        }
        else if(value < distribution.z)
        {
            return Mathf.InverseLerp(distribution.y, distribution.z, value);
        }
        else
        {
            return 1;
        }
    }

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
        landUsageBuffer.Dispose();

        foreach (var gene in population)
        {
            gene.centroids.Dispose();
        }

        DisposeForBatching();
    }
}
