import java.io.*;
import java.text.DecimalFormat;
public class Main {
	public static void main(String[] args) throws IOException, InterruptedException {
		int numOfExecutions = 50;
		DecimalFormat df = new DecimalFormat("#.###");
		int fileNum = 8;
		int verNum = 2;
		double rewire = 0.15;
		double beta = 0.3151;
		String name = "hcg-experiment"
		while(rewire>=0.0) {
			double delta = 0.589;
			while(delta>=0.289) {
				System.out.println("python ../../SEIDRmain2.py " +df.format(delta)+" "+beta+" "+rewire+" "+fileNum+ " "+verNum);
				BufferedWriter bw = new BufferedWriter(new FileWriter(new File("hcg-experiment"+fileNum+"."+verNum+".csv")));
		//		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("experiment-beta9.2.csv")));
				bw.write("p_edge_creation,p_infected,gamma,beta,delta,epsilon,zeta,eta,N,elapsed_time,timesteps,events,timesteps_with_events,mean_outbreak_size,max_outbreak_size,max_outbreak_proportion,exposed_from_infected,exposed_from_dead,rewire_degree\n");
		//		bw.write("p_edge_creation,p_infected,gamma,beta,eta,N,elapsed_time,timesteps,events,timesteps_with_events,mean_outbreak_size,max_outbreak_size,max_outbreak_proportion,exposed_from_infected\n");
				bw.close();
				for (int n = 0; n < numOfExecutions; n++) {
					Runtime rt = Runtime.getRuntime();
					Process pr = rt.exec("python ../../SEIDRmain2.py " +df.format(delta)+" "+beta+" "+rewire+" "+fileNum+ " "+verNum+ " "+ name);
		//			Process pr = rt.exec("python ../../SEIRinfluenza.py");
					System.out.println(pr.waitFor());
				}
				delta=delta-0.1;
				fileNum-=1;
			}
			rewire=rewire-0.15;
		}

	}
}
