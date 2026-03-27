

open("BatteryProfile.mat");
s=serialport("COM6",115200);
configureTerminator(s,"LF");
flush(s);

I=prf.I;
U=prf.U;
SoH=prf.SoH;
SoC=prf.SoC;
soc_ekf=nan(size(I));
u_ekf=nan(size(I));

Ts=0.1;

figure;
h = plot(NaN,NaN,'-o');
xlabel("time step");
ylabel("output");
grid on;


for k=1:length(I)
    t_start = tic; 

    
    writeline(s, sprintf('%.6f,%.6f,%.6f', I(k), U(k),SoH(k)));

    response = readline(s);
    data=sscanf(response,"%f,%f");

    soc_ekf(k)=data(1);
    u_ekf(k)=data(2);

    set(h,'XData',1:k,'YData',soc_ekf(1:k));
    drawnow limitrate;

   
    elapsed = toc(t_start);
    pause(max(0, Ts - elapsed));

end


clear s


