function plot_seir()
    clear all; close all;

    t_start = 0;
    t_end = 30;

    beta = 0.001151;
    gamma = 0.06851662;
    eta = 0.083333;
    s0 = 5000;
    e0 = 0;
    i0 = 10;
    r0 = 0;

    [T, Y] = ode45(@seir, [t_start, t_end], [s0, e0, i0, r0]);
    plot(T, Y(:,1),'-', T,Y(:,2),'-.', T,Y(:,3),'+', T,Y(:,4),'x');


    function dy = seir(t,y)
        dy=zeros(4,1);
        dy(1) = -(beta*y(1)*y(3));
        dy(2) = (beta*y(1)*y(3))-(eta*y(2));
        dy(3) = (eta*y(2))-(gamma*y(3));
        dy(4) = (gamma*y(3));
    end
end

