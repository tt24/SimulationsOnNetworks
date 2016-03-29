function plot_seir()
    clear all; close all;

    t_start = 0;
    t_end =120;

    beta = 0.001151;
    gamma = 0.06851662;
    eta = 0.083333;
    s0 = 4999;
    e0 = 0;
    i0 = 1;
    r0 = 0;

    [T, Y] = ode45(@seir, [t_start, t_end], [s0, e0, i0, r0]);
    figure
    plot(T, Y(:,1),'y-', T,Y(:,2),'b.', T,Y(:,3),'r-.', T,Y(:,4),'g--');
    xlabel('Time steps');
    ylabel('Number of individuals');
    legend('susceptible','exposed','infected','removed')
    EI=plus(Y(:,2),Y(:,3));
    maxEI=max(EI);
    disp(maxEI);
    i=find(EI==maxEI);
    disp(i);
    disp(T(i));
    disp(Y(i,1));
    disp(Y(i+1,1));

    function dy = seir(t,y)
        dy=zeros(4,1);
        dy(1) = -(beta*y(1)*y(3));
        dy(2) = (beta*y(1)*y(3))-(eta*y(2));
        dy(3) = (eta*y(2))-(gamma*y(3));
        dy(4) = (gamma*y(3));
    end
end
