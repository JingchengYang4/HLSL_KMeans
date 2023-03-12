using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

[BurstCompile]
public struct AhpIndexJob : IJobParallelFor
{
    [ReadOnly]
    public NativeArray<int> landUsage;
    
    [ReadOnly]
    public NativeArray<int> buildingAreas;

    [NativeDisableParallelForRestriction] 
    [WriteOnly]
    public NativeArray<float> indices;

    [ReadOnly] 
    public NativeArray<int> coverTypeArea;

    [ReadOnly] 
    public int buildingCount;

    public void Execute(int startIndex)
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

            int buildingArea = buildingAreas[startIndex*buildingCount+i];
            
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
        
        float environmentHarmIndex = 0;
        environmentHarmIndex += landUsage[startIndex*4+0] * 1.0f / coverTypeArea[0] * 4;
        environmentHarmIndex += landUsage[startIndex*4+2] * 1.0f / coverTypeArea[2] * 3;
        environmentHarmIndex += landUsage[startIndex*4+1] * 1.0f / coverTypeArea[1] * 2;
        environmentHarmIndex /= 9f;

        /*for (int i = 0; i < 4; i++)
        {
            Debug.Log($"{landUsage[i]} / {coverTypeArea[i]}");
        }*/
        
        float profitIndex = EvaluateDistribution(profits, 83110.52f, 153891.6f, 352343.6f);
        float constructionCostIndex = EvaluateDistribution(costs, 782594.3f, 1085982f, 1454249f);
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

        indices[startIndex] = index;
    }
    
    float PaybackIndex(float paybackYear)
    {
        return 1 - 1 / (1 + math.exp(-(3f / 10f * paybackYear + 4.5f)));
    }

    float DevelopmentDurationIndex(float constructionDuration)
    {
        return math.exp(-0.6f * constructionDuration);
    }

    float EvaluateDistribution(float value, float q1, float q3, float max)
    {
        if (value < q1)
        {
            return math.unlerp(0, q1, value);
        }
        else if (value < q3)
        {
            return math.unlerp(q1, q3, value);
        }
        else if(value < max)
        {
            return math.unlerp(q3, max, value);
        }
        else
        {
            return 1;
        }
    }
}