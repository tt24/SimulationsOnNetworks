import java.io.*;

public class Main {
	public static void main(String[] args) throws IOException, InterruptedException {
		int numOfExecutions = 50;
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("seir-experiment-influenza1.1.csv")));
//		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("experiment-beta9.2.csv")));
//		bw.write("p_edge_creation,p_infected,gamma,beta,delta,epsilon,zeta,eta,N,elapsed_time,timesteps,events,timesteps_with_events,mean_outbreak_size,max_outbreak_size,max_outbreak_proportion,exposed_from_infected,exposed_from_dead,rewire_degree\n");
		bw.write("p_edge_creation,p_infected,gamma,beta,eta,N,elapsed_time,timesteps,events,timesteps_with_events,mean_outbreak_size,max_outbreak_size,max_outbreak_proportion,exposed_from_infected\n");
		bw.close();
		for (int n = 0; n < numOfExecutions; n++) {
			Runtime rt = Runtime.getRuntime();
//			Process pr = rt.exec("python ../../SEIDRmain.py");
			Process pr = rt.exec("python ../../SEIRinfluenza.py");
			System.out.println(pr.waitFor());
		}
	}
}
