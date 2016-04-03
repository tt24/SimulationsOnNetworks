import java.io.*;
import java.text.DecimalFormat;
public class Main {
	public static void main(String[] args) throws IOException, InterruptedException {
		int numOfExecutions = 50;
		DecimalFormat df = new DecimalFormat("#.###");
		double[] deltas = {0.289,0.389,0.489,0.589};
		double[] betas = {0.1151, 0.2151};
		double[] rewires = {0.0,0.15,0.25,0.35};
		int beta = 1;
		while (beta<=1) {
			int delta = 3;
			while(delta>=2) {
				int rewire = 0;
				while(rewire<=3) {
					String name = "HCgraphExperiment/temp/beta"+beta+"/delta"+delta+"/experiment"+rewire;
					System.out.println("python ../../SEIDRmain2.py " +name+ " "+deltas[delta]+" "+betas[beta]+" "+rewires[rewire]);
					BufferedWriter bw = new BufferedWriter(new FileWriter(new File(name+".csv")));
			//		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("experiment-beta9.2.csv")));
					bw.write("p_edge_creation,p_infected,gamma,beta,delta,epsilon,zeta,eta,N,elapsed_time,timesteps,events,timesteps_with_events,mean_outbreak_size,max_outbreak_size,max_outbreak_proportion,exposed_from_infected,exposed_from_dead,rewire_degree\n");
			//		bw.write("p_edge_creation,p_infected,gamma,beta,eta,N,elapsed_time,timesteps,events,timesteps_with_events,mean_outbreak_size,max_outbreak_size,max_outbreak_proportion,exposed_from_infected\n");
					bw.close();
					for (int n = 0; n < numOfExecutions; n++) {
						System.out.println(n);
						Runtime rt = Runtime.getRuntime();
						Process pr = rt.exec("python ../../SEIDRmain2.py " +deltas[delta]+" "+betas[beta]+" "+rewires[rewire]+" "+ name);
			//			Process pr = rt.exec("python ../../SEIRinfluenza.py");
						System.out.println(pr.waitFor());
					}
					rewire++;
				}
				delta--;
			}
			beta++;
		}

	}
}
