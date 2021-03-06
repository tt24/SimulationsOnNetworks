function plot_seidr()
    clear all; close all;

    t_start = 0;
    t_end = 70;

    beta = 0.0001151;
    gamma = 0.16851662;
    eta = 0.083333;
    epsilon = 0.2;
    zeta = 0.5;
    delta = 0.0189;
    
    disp(gamma*epsilon/beta);
    disp(zeta/delta);
    
    s0 = 250;
    e0 = 0;
    i0 = 1;
    d0 = 0;
    r0 = 750;

    [T, Y] = ode45(@seidr, [t_start, t_end], [s0, e0, i0, d0, r0]);
    plot(T, Y(:,1),'y-', T,Y(:,2),'b.', T,Y(:,3),'r.-', T,Y(:,4),'black-', T,Y(:,5),'g--');
    xlabel('Time steps');
    ylabel('Number of individuals');
    legend('susceptible','exposed','infected','dead','removed')
    
    EI=plus(Y(:,2),Y(:,3));
    EID=plus(EI,Y(:,4));
    maxEID=max(EID);
    disp(maxEID);
    i=find(EID==maxEID);
    disp(i);
    disp(T(i));
    disp(Y(i,1));
    disp(Y(i+1,1));
    disp(Y(i,3)*(beta*Y(i,1)-gamma*epsilon)+Y(i,4)*(delta*Y(i,1)-zeta));
    disp(Y(i+1,3)*(beta*Y(i+1,1)-gamma*epsilon));
    disp(Y(i+1,4)*(delta*Y(i+1,1)-zeta));
    disp(Y(size(Y(:,1)),1));


    function dy = seidr(t,y)
        dy=zeros(5,1);
        dy(1) = -(beta*y(1)*y(3))-(delta*y(1)*y(4));
        dy(2) = (beta*y(1)*y(3))+(delta*y(1)*y(4))-(eta*y(2));
        dy(3) = (eta*y(2))-(gamma*y(3));
        dy(4) = ((1-epsilon)*gamma*y(3))-(zeta*y(4));
        dy(5) = (epsilon*gamma*y(3))+(zeta*y(4));
        
    end
end