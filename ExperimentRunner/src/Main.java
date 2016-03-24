import java.io.*;

public class Main {
	public static void main(String[] args) throws IOException, InterruptedException {
		int numOfExecutions = 50;
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("experiment10.csv")));
		bw.write("pInfected, gamma, beta, eta, N, elapsed_time, timesteps, events, timesteps_with_events, mean_outbreak_size, max_outbreak_size, max_outbreak_proportion\n");
		bw.close();
		for (int n = 0; n < numOfExecutions; n++) {
			Runtime rt = Runtime.getRuntime();
			Process pr = rt.exec("python ../../SEIDRmain.py");
			System.out.println(pr.waitFor());
		}
	}
}
