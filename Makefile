USERNAME=""
ifeq ($(strip $(USERNAME)), "")
	BASE_TAG=""
else
	BASE_TAG=$(USERNAME)/
endif

docker-build:
	docker build -t $(BASE_TAG)chiron:latest-py3-gpu -f Dockerfile.py3.gpu . 
	docker build -t $(BASE_TAG)chiron:latest-py3 -f Dockerfile.py3.cpu . 

.PHONY: docker-build

docker-push: docker-build
	docker push $(BASE_TAG)chiron:latest-py3-gpu
	docker push $(BASE_TAG)chiron:latest-py3

.PHONY: docker-push
	
	
