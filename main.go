package main

import (
    "fmt"
    "github.com/milvus-io/milvus-sdk-go/milvus"
)

func main() {
    // 创建 Milvus 客户端
    client, err := milvus.NewClient(
        "127.0.0.1",
        "19530",
        milvus.WithDialTimeout(10*time.Second),
    )
    if err != nil {
        panic(err)
    }

    // 创建集合
    collection, err := CreateCollection(client, "my_collection", 128)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Created collection: %s\n", collection.Name)

    // 创建分区
    partition, err := CreatePartition(client, collection.Name, "my_partition", []string{"age"})
    if err != nil {
        panic(err)
    }
    fmt.Printf("Created partition: %s\n", partition.Name)

    // 插入向量
    vectors := [][]float32{
        {0.1, 0.2, 0.3, 0.4},
        {0.2, 0.3, 0.4, 0.5},
        {0.3, 0.4, 0.5, 0.6},
    }
    vectorIDs := []int64{1, 2, 3}
    err = InsertVectors(client, collection.Name, partition.Name, vectors, vectorIDs)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Inserted vectors with IDs: %v\n", vectorIDs)

    // 搜索向量
    searchVectors := [][]float32{
        {0.15, 0.25, 0.35, 0.45},
    }
    searchParam := milvus.SearchParam{
        CollectionName: collection.Name,
        PartitionNames: []string{partition.Name},
        QueryVectors:   searchVectors,
        TopK:           10,
    }
    res, err := SearchVectors(client, searchParam)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Search results: %v\n", res)

    // 删除向量
    err = DeleteVectorsByID(client, collection.Name, []int64{1, 2, 3})
    if err != nil {
        panic(err)
    }
    fmt.Printf("Deleted vectors with IDs: %v\n", []int64{1, 2, 3})

    // 删除分区
    err = DropPartition(client, collection.Name, partition.Name)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Dropped partition: %s\n", partition.Name)

    // 删除集合
    err = DropCollection(client, collection.Name)
    if err != nil {
        panic(err)
    }
    fmt.Printf("Dropped collection: %s\n", collection.Name)

    // 关闭客户端
    err = client.Close()
    if err != nil {
        panic(err)
    }
}

// 创建集合
func CreateCollection(client milvus.MilvusClient, name string, dim int) (*milvus.Collection, error) {
    schema := milvus.CollectionSchema{
        Name:      name,
        Dimension: dim,
    }
    return client.CreateCollection(schema)
}

// 创建分区
func CreatePartition(client milvus.MilvusClient, collectionName string, partitionName string, partitionFields []string) (*milvus.Partition, error) {
    return client.CreatePartition(collectionName, partitionName, partitionFields)
}

// 插入向量
func InsertVectors(client milvus.MilvusClient, collectionName string, partitionName string, vectors [][]float32, vectorIDs []int64) error {
    _, err := client.Insert(collectionName, partitionName, vectors, vectorIDs)
    return err
}

// 搜索向量
func SearchVectors(client milvus.MilvusClient, searchParam milvus.SearchParam) ([]milvus.SearchResult, error) {
    return client.Search(searchParam)
}

// 删除向量
func DeleteVectorsByID(client milvus.MilvusClient, collectionName string, vectorIDs []int64) error {
    _, err := client.DeleteByID(collectionName, vectorIDs)
    return err
}

// 删除分区
func DropPartition(client milvus.MilvusClient, collectionName string, partitionName string) error {
    _, err := client.DropPartition(collectionName, partitionName)
    return err
}

// 删除集合
func DropCollection(client milvus.MilvusClient, collectionName string) error {
    _, err := client.DropCollection(collectionName)
    return err
}
