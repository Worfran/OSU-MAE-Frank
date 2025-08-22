function rate_loop
a=5e-10;
S0=a.^3/4; % fcc
Z=84;
alpha=Z*S0/a.^2;

T=600; %K
kb=8.62e-5; % eV/K
Ei=1.2; % eV
Ev=1.8;% eV
nu=2e13; % 1/s
wi=nu*exp(-Ei/kb./T);
wv=nu*exp(-Ev/kb./T);
Di=a.^2.*wi; % fcc
Dv=a.^2.*wv; % fcc
    
K0=1e-6./S0;
% dislocation
NL=1e-7/S0;
b=a./sqrt(3); %(111)
r0=4.*a; 
rL_0=a;


[t1,y1]=ode15s(@diffeq,[0 1e5],[0 0 rL_0]);%
subplot(2,1,1)
loglog(t1,y1(:,1:2).*S0,'LineWidth',2);
xlabel('time');ylabel('concnetration');
legend('Interstitial','Vacancy');
subplot(2,1,2)
plot(t1,y1(:,3)*1e9,'LineWidth',2);
xlabel('time');ylabel('radius (nm)');
%[t2,y2]=ode15s(@diffeq,[0 5e8],[0 0]);
%loglog(t1,y1,t2,y2,'--');
% xlabel('time');ylabel('concnetration');
% legend('Vacancy','Interstitial');
    function dy=diffeq(t,y)
        %gl=0;
        dy=[0  0 0]';
        Kiv=alpha*(Di+Dv);
        Kis=Di./log(8.*y(3)./r0).*2*pi.^2.*y(3);
        Kvs=Dv./log(8.*y(3)./r0).*2*pi.^2.*y(3);
        dy(1)=K0-Kiv*y(1)*y(2)-Kis*y(1)*NL;
        dy(2)=K0-Kiv*y(1)*y(2)-Kvs*y(2)*NL;
        dy(3)=S0.*pi./b./log(8.*y(3)./r0).*(Di.*y(1)-Dv.*y(2));
    end
end
