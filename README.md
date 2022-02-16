# voice_assistant

Réseau d'experts en automobile

# Development
## Prerequisites
### Install Docker
More info at https://docs.docker.com/engine/install

### Install Docker Compose
More info at https://docs.docker.com/compose/install

### Install make
```
sudo apt-get install build-essential
```

## Install project

### Start project
```shell
make docker-up
```

### Import data
```shell
docker cp dump.sql lideo-caselawanalytics-com-mysql:/tmp
make docker-exec CONTAINER=mysql
mysql -u lideo -p lideo < /tmp/dump.sql
```

### Access the app
Browse http://127.0.0.1:5000/docs


# Deployment
## Prerequisites

```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu hirsute stable"
sudo apt update
sudo apt install apt-transport-https software-properties-common python3-pip docker-ce
sudo usermod -aG docker ${USER}
# Add "Defaults env_keep += "SSH_AUTH_SOCK"" in /etc/sudoers
sudo pip install docker-compose
# Add "DOCKER_ENV=prod" in /etc/environment
# Add "MYSQL_ROOT_PASSWORD=<db password>" in /etc/environment
# Add "MYSQL_PASSWORD=<db password>" in /etc/environment
sudo mkdir -p /srv && cd /srv
sudo git clone git@github.com:case-law-analytics/lideo.caselawanalytics.com.git
```
