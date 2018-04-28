USERNAME=""
TAG="latest-py3"

ifeq ($(strip $(USERNAME)), "")
	BASE_TAG=""
else
	BASE_TAG=$(USERNAME)/
endif

docker-build:
	docker build -t $(BASE_TAG)chiron:$(TAG)-gpu -f Dockerfile.py3.gpu . 
	docker build -t $(BASE_TAG)chiron:$(TAG) -f Dockerfile.py3.cpu . 

.PHONY: docker-build

docker-push: docker-build
	docker push $(BASE_TAG)chiron:$(TAG)-gpu
	docker push $(BASE_TAG)chiron:$(TAG)

.PHONY: docker-push
	
	
