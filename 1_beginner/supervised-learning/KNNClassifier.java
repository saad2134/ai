import java.util.*;

public class KNNClassifier {
    private double[][] trainingData;
    private int[] trainingLabels;
    
    public void fit(double[][] data, int[] labels) {
        this.trainingData = data;
        this.trainingLabels = labels;
    }
    
    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }
    
    public int predict(double[] sample, int k) {
        // Calculate distances to all training samples
        double[][] distances = new double[trainingData.length][2];
        
        for (int i = 0; i < trainingData.length; i++) {
            double distance = euclideanDistance(trainingData[i], sample);
            distances[i][0] = distance;
            distances[i][1] = trainingLabels[i];
        }
        
        // Sort by distance
        Arrays.sort(distances, (a, b) -> Double.compare(a[0], b[0]));
        
        // Count votes from k nearest neighbors
        Map<Integer, Integer> voteCount = new HashMap<>();
        for (int i = 0; i < k; i++) {
            int label = (int) distances[i][1];
            voteCount.put(label, voteCount.getOrDefault(label, 0) + 1);
        }
        
        // Return majority vote
        return Collections.max(voteCount.entrySet(), Map.Entry.comparingByValue()).getKey();
    }
    
    public static void main(String[] args) {
        // Sample data: [age, income] -> credit_approval (0=denied, 1=approved)
        double[][] features = {
            {25, 30000}, {35, 50000}, {45, 80000}, {55, 60000},
            {30, 35000}, {40, 70000}, {50, 90000}, {60, 95000}
        };
        
        int[] labels = {0, 1, 1, 1, 0, 1, 1, 1}; // 0=denied, 1=approved
        
        KNNClassifier knn = new KNNClassifier();
        knn.fit(features, labels);
        
        System.out.println("=== K-NEAREST NEIGHBORS ===");
        
        double[] newApplicant = {38, 55000};
        int prediction = knn.predict(newApplicant, 3);
        
        System.out.printf("Applicant [age=%.0f, income=%.0f] prediction: %s%n",
            newApplicant[0], newApplicant[1], prediction == 1 ? "APPROVED" : "DENIED");
    }
}