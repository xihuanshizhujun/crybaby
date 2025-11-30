import weaviate

try:
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051,
    )



    # client.collections.delete(
    #     #     "FinancialDoc"
    #     # )
    #
    collection = client.collections.get("FinancialDoc")
    for item in collection.iterator(
            include_vector=True
    ):
        print(item.properties['content'])
        # print(item.vector)
finally:
    client.close()