using System.Collections.Generic;
using UnityEngine;

public static class Utilities
{
    public static float StdDev(this IEnumerable<float> values)
    {
        // ref: http://warrenseen.com/blog/2006/03/13/how-to-calculate-standard-deviation/
        float mean = 0;
        float sum = 0;
        float stdDev = 0;
        int n = 0;
        foreach (float val in values)
        {
            n++;
            float delta = val - mean;
            mean += delta / n;
            sum += delta * (val - mean);
        }
        if (1 < n)
            stdDev = Mathf.Sqrt(sum / (n - 1));

        return stdDev;
    }
}