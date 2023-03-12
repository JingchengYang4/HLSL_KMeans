public struct Center
{
    public int x;
    public int y;
    
    public int buildingType;

    public int totalX;
    public int totalY;

    public int cumX;
    public int cumY;

    public Center(int _x, int _y, int _buildingType)
    {
        x = _x;
        y = _y;
        totalX = 0;
        totalY = 0;
        cumX = 0;
        cumY = 0;
        buildingType = _buildingType;
    }
}