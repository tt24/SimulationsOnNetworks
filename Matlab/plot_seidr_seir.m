function plot_seidr_seir()
    clear all; close all;

    t_start = 0;
    t_end = 70;

    beta = 0.001151;
    gamma = 0.16851662;
    eta = 0.083333;
    epsilon = 0.2;
    zeta = 0.5;
    delta = 0.001151;
    
    disp(gamma*epsilon/beta);
    disp(zeta/delta);
    
    s0 = 4999;
    e0 = 0;
    i0 = 1;
    d0 = 0;
    r0 = 0;
    
    [T1, Y1] = ode45(@seir, [t_start, t_end], [s0, e0, i0, r0]);
    plot(T1, Y1(:,1),'r-');
    hold on
    
    while delta < 0.04189
        [T, Y] = ode45(@seidr, [t_start, t_end], [s0, e0, i0, d0, r0]);
        plot(T, Y(:,1),'k-');
        delta=delta+0.005;
    end
    xlabel('Time steps');
    ylabel('Number of individuals');
    legend('seir S','seidr S');
    hold off
    
    delta = 0.001151;
    [T1, Y1] = ode45(@seir, [t_start, t_end], [s0, e0, i0, r0]);
    EI1=plus(Y1(:,2),Y1(:,3));
    plot(T1, EI1,'r-');
    hold on
    
    while delta < 0.04189
        [T, Y] = ode45(@seidr, [t_start, t_end], [s0, e0, i0, d0, r0]);
        EI=plus(Y(:,2),Y(:,3));
        EID=plus(EI,Y(:,4));
        plot(T, EID,'k-');
        delta=delta+0.005;
    end
    xlabel('Time steps');
    ylabel('Number of individuals');
    legend('seir E+I','seidr E+I+D');
    hold off

    delta = 0.001151;
    [T1, Y1] = ode45(@seir, [t_start, t_end], [s0, e0, i0, r0]);
    plot(T1, Y1(:,1),'r-');
    hold on
    
    while zeta > 0.1
        [T, Y] = ode45(@seidr, [t_start, t_end], [s0, e0, i0, d0, r0]);
        plot(T, Y(:,1),'k-');
        zeta=zeta-0.1;
    end
    xlabel('Time steps');
    ylabel('Number of individuals');
    legend('seir S','seidr S');
    hold off
    
    
    
    zeta = 0.5;
    [T1, Y1] = ode45(@seir, [t_start, t_end], [s0, e0, i0, r0]);
    EI1=plus(Y1(:,2),Y1(:,3));
    plot(T1, EI1,'r-');
    hold on
    
    while zeta > 0.1
        [T, Y] = ode45(@seidr, [t_start, t_end], [s0, e0, i0, d0, r0]);
        EI=plus(Y(:,2),Y(:,3));
        EID=plus(EI,Y(:,4));
        plot(T, EID,'k-');
        zeta=zeta-0.1;
    end
    xlabel('Time steps');
    ylabel('Number of individuals');
    legend('seir E+I','seidr E+I+D');
    hold off
    
    

    zeta = 0.5;
    [T1, Y1] = ode45(@seir, [t_start, t_end], [s0, e0, i0, r0]);
    EI1=plus(Y1(:,2),Y1(:,3));
    plot(T1, EI1,'r-');
    hold on
    
    while epsilon > 0.01
        [T, Y] = ode45(@seidr, [t_start, t_end], [s0, e0, i0, d0, r0]);
        EI=plus(Y(:,2),Y(:,3));
        EID=plus(EI,Y(:,4));
        plot(T, EID,'k-');
        epsilon=epsilon-0.05;
    end
    xlabel('Time steps');
    ylabel('Number of individuals');
    legend('seir E+I','seidr E+I+D');
    hold off
    
    epsilon = 0.2;
    t_end = 20;
    [T1, Y1] = ode45(@seir, [t_start, t_end], [s0, e0, i0, r0]);
    plot(T1, Y1(:,1),'r-');
    hold on
    
    while epsilon > 0.01
        [T, Y] = ode45(@seidr, [t_start, t_end], [s0, e0, i0, d0, r0]);
        plot(T, Y(:,1),'k-');
        epsilon=epsilon-0.05;
    end
    xlabel('Time steps');
    ylabel('Number of individuals');
    legend('seir S','seidr S');
    hold off
    
    
    function dy = seidr(t,y)
        dy=zeros(5,1);
        dy(1) = -(beta*y(1)*y(3))-(delta*y(1)*y(4));
        dy(2) = (beta*y(1)*y(3))+(delta*y(1)*y(4))-(eta*y(2));
        dy(3) = (eta*y(2))-(gamma*y(3));
        dy(4) = ((1-epsilon)*gamma*y(3))-(zeta*y(4));
        dy(5) = (epsilon*gamma*y(3))+(zeta*y(4));
        
    end

    function dy = seir(t,y)
        dy=zeros(4,1);
        dy(1) = -(beta*y(1)*y(3));
        dy(2) = (beta*y(1)*y(3))-(eta*y(2));
        dy(3) = (eta*y(2))-(gamma*y(3));
        dy(4) = (gamma*y(3));
    end
end