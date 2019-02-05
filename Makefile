IMAGE_NAME=c3d_sport1m

build:
	docker build -t $(IMAGE_NAME) .

dev:
	docker run --rm -ti  \
		-v $(PWD)/:/project \
		-w '/project' \
		$(IMAGE_NAME)

example:
	docker run --rm -ti  \
		-v $(PWD)/:/project \
		-w '/project' \
		$(IMAGE_NAME) python3 c3d/example.py