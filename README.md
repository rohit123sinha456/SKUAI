Install nvidia container toolking
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker 
 
 
 docker compose build
 sudo chown -R 1001:1001 annotation/
 sudo chown -R 1001:1001 aiapp/
 generate token from label studio
 place it in .env file
 then docker compose build
 docker compose up- d