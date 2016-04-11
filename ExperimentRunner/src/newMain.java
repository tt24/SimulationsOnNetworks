import java.io.*;

public class newMain {
	public static void main(String[] args) throws IOException, InterruptedException {
		int numOfExecutions = 50;
		double[] betas = {0.1151, 0.2151, 0.3151};
		int beta = 0;
		//while(beta<=2) {
			String filename = "seidr-experiment-influenza-er1.1";
			BufferedWriter bw = new BufferedWriter(new FileWriter(new File(filename+".csv")));
			bw.write("p_edge_creation,p_infected,gamma,beta,delta,epsilon,zeta,eta,N,elapsed_time,timesteps,events,timesteps_with_events,mean_outbreak_size,max_outbreak_size,max_outbreak_proportion,exposed_from_infected,exposed_from_dead,rewire_degree\n");
			bw.close();
			for (int n = 0; n < numOfExecutions; n++) {
				Runtime rt = Runtime.getRuntime();
				Process pr = rt.exec("python ../../SEIDRinfluenza.py");
				System.out.println(n+" "+pr.waitFor());
			}
			beta++;
		//}
	}
}
