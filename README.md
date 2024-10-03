Install nvidia container toolking
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker 
 
If needed Install 
https://github.com/docker/compose/issues/8142
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version
 
 docker compose build
 sudo chown -R 1001:1001 annotation/
 sudo chown -R 1001:1001 aiapp/
 generate token from label studio
 place it in .env file
 then docker compose build
 docker compose up- d

 SV1
 {
        "0" : {
            "Name" : "Fax",
            "Type" : "Text"
        },
        "1" : {
            "Name" : "Name",
            "Type" : "Text"
        },
        "2" : {
            "Name" : "PODate",
            "Type" : "Text"
        },
        "3" : {
            "Name" : "PONumber",
            "Type" : "Text"
        },
        "4" : {
            "Name" : "Supplier",
            "Type" : "Text"
        },
        "5" : {
            "Name" : "TableContents",
            "Type" : "Table"
        },
        "6" : {
            "Name" : "Telefone",
            "Type" : "Text"
        }
    }

