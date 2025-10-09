import java.util.Arrays;

public class LinearRegression {
    private double slope;
    private double intercept;
    
    public void train(double[] x, double[] y) {
        int n = x.length;
        double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
        
        for (int i = 0; i < n; i++) {
            sumX += x[i];
            sumY += y[i];
            sumXY += x[i] * y[i];
            sumXX += x[i] * x[i];
        }
        
        this.slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        this.intercept = (sumY - this.slope * sumX) / n;
    }
    
    public double predict(double x) {
        return slope * x + intercept;
    }
    
    public static void main(String[] args) {
        // Sample data: house size vs price
        double[] sizes = {1000, 1500, 2000, 2500, 3000};
        double[] prices = {300000, 400000, 500000, 550000, 600000};
        
        LinearRegression model = new LinearRegression();
        model.train(sizes, prices);
        
        System.out.println("=== LINEAR REGRESSION ===");
        System.out.printf("Model: price = %.2f * size + %.2f%n", model.slope, model.intercept);
        
        double newSize = 2800;
        double predictedPrice = model.predict(newSize);
        System.out.printf("Predicted price for %.0f sqft: $%.2f%n", newSize, predictedPrice);
    }
}