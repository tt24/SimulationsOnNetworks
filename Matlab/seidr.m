function plot_seidr()
    clear all; close all;

    t_start = 0;
    t_end = 30;

    beta = 0.001151;
    gamma = 0.06851662;
    eta = 0.083333;
    epsilon = 0.2;
    zeta = 0.5;
    delta = 0.389;
    
    s0 = 5000;
    e0 = 0;
    i0 = 10;
    d0 = 0;
    r0 = 0;

    [T, Y] = ode45(@seidr, [t_start, t_end], [s0, e0, i0, d0, r0]);
    plot(T, Y(:,1),'-', T,Y(:,2),'-.', T,Y(:,3),'+', T,Y(:,4),'x', T,Y(:,5),'.');


    function dy = seidr(t,y)
        dy=zeros(5,1);
        dy(1) = -(beta*y(1)*y(3))-(delta*y(1)*y(5));
        dy(2) = (beta*y(1)*y(3))+(delta*y(1)*y(5))-(eta*y(2));
        dy(3) = (eta*y(2))-(gamma*y(3));
        dy(4) = (epsilon*gamma*y(3))+(zeta*y(5));
        dy(5) = ((1-epsilon)*gamma*y(3))-(zeta*y(5));
    end
end