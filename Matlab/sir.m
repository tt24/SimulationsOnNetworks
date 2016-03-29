function sir()

function dy = sir_de(t,y)
dy = zeros(3,1);
dy(1) = -0.00218*y(1)*y(2);
dy(2) = 0.00218*y(1)*y(2)-0.44036*y(2);
dy(3) = 0.44036*y(2);
end

[T,Y] = ode45(@sir_de, [0, 14], [762, 1, 0]);
figure
plot(T, Y(:,1),'y-', T,Y(:,2),'r-.', T,Y(:,3),'g--');
xlabel('Time (days)');
ylabel('Number of boys');
legend('susceptible','infected','removed')
end

