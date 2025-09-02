"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Slider } from "@/components/ui/slider"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Card, CardContent, CardDescription, CardHeader } from "@/components/ui/card"

interface CreateKnowledgeBaseDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function CreateKnowledgeBaseDialog({ open, onOpenChange }: CreateKnowledgeBaseDialogProps) {
  const [currentStep, setCurrentStep] = useState("basic")
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    tags: "",
    embeddingModel: "",
    retrievalMode: "vector",
    topK: [3],
    scoreThreshold: [0.5],
    hybridWeight: [0.7],
    rerankModel: "",
  })

  const handleCreate = () => {
    // Handle knowledge base creation
    console.log("Creating knowledge base:", formData)
    onOpenChange(false)
    // Reset form
    setFormData({
      name: "",
      description: "",
      tags: "",
      embeddingModel: "",
      retrievalMode: "vector",
      topK: [3],
      scoreThreshold: [0.5],
      hybridWeight: [0.7],
      rerankModel: "",
    })
    setCurrentStep("basic")
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>新建知识库</DialogTitle>
          <DialogDescription>创建一个新的知识库来管理您的文档和知识内容</DialogDescription>
        </DialogHeader>

        <Tabs value={currentStep} onValueChange={setCurrentStep} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="basic">基本信息</TabsTrigger>
            <TabsTrigger value="index">索引策略</TabsTrigger>
            <TabsTrigger value="retrieval">检索默认</TabsTrigger>
          </TabsList>

          <TabsContent value="basic" className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="name">知识库名称 *</Label>
              <Input
                id="name"
                placeholder="输入知识库名称"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="description">描述</Label>
              <Textarea
                id="description"
                placeholder="描述这个知识库的用途和内容"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="tags">标签（可选）</Label>
              <Input
                id="tags"
                placeholder="用逗号分隔多个标签"
                value={formData.tags}
                onChange={(e) => setFormData({ ...formData, tags: e.target.value })}
              />
            </div>
          </TabsContent>

          <TabsContent value="index" className="space-y-4">
            <div className="space-y-2">
              <Label>Embedding 模型 *</Label>
              <Select
                value={formData.embeddingModel}
                onValueChange={(value) => setFormData({ ...formData, embeddingModel: value })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="选择 Embedding 模型" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="text-embedding-3-large">text-embedding-3-large</SelectItem>
                  <SelectItem value="text-embedding-3-small">text-embedding-3-small</SelectItem>
                  <SelectItem value="text-embedding-ada-002">text-embedding-ada-002</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </TabsContent>

          <TabsContent value="retrieval" className="space-y-6">
            <div className="space-y-4">
              <Label>检索模式</Label>
              <RadioGroup
                value={formData.retrievalMode}
                onValueChange={(value) => setFormData({ ...formData, retrievalMode: value })}
              >
                <div className="grid gap-4">
                  <Card className={formData.retrievalMode === "vector" ? "ring-2 ring-primary" : ""}>
                    <CardHeader className="pb-2">
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="vector" id="vector" />
                        <Label htmlFor="vector" className="font-medium">
                          向量检索
                        </Label>
                      </div>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <CardDescription>基于语义相似度的向量检索</CardDescription>
                    </CardContent>
                  </Card>

                  <Card className={formData.retrievalMode === "fulltext" ? "ring-2 ring-primary" : ""}>
                    <CardHeader className="pb-2">
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="fulltext" id="fulltext" />
                        <Label htmlFor="fulltext" className="font-medium">
                          全文检索
                        </Label>
                      </div>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <CardDescription>基于关键词匹配的全文检索</CardDescription>
                    </CardContent>
                  </Card>

                  <Card className={formData.retrievalMode === "hybrid-weight" ? "ring-2 ring-primary" : ""}>
                    <CardHeader className="pb-2">
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="hybrid-weight" id="hybrid-weight" />
                        <Label htmlFor="hybrid-weight" className="font-medium">
                          混合-权重
                        </Label>
                      </div>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <CardDescription>结合向量和全文检索，按权重融合</CardDescription>
                    </CardContent>
                  </Card>

                  <Card className={formData.retrievalMode === "hybrid-rerank" ? "ring-2 ring-primary" : ""}>
                    <CardHeader className="pb-2">
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="hybrid-rerank" id="hybrid-rerank" />
                        <Label htmlFor="hybrid-rerank" className="font-medium">
                          混合-Rerank
                        </Label>
                      </div>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <CardDescription>使用 Rerank 模型对混合结果重新排序</CardDescription>
                    </CardContent>
                  </Card>
                </div>
              </RadioGroup>
            </div>

            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Top K: {formData.topK[0]}</Label>
                <Slider
                  value={formData.topK}
                  onValueChange={(value) => setFormData({ ...formData, topK: value })}
                  max={20}
                  min={1}
                  step={1}
                  className="w-full"
                />
              </div>

              {(formData.retrievalMode === "vector" || formData.retrievalMode.includes("hybrid")) && (
                <div className="space-y-2">
                  <Label>Score 阈值: {formData.scoreThreshold[0]}</Label>
                  <Slider
                    value={formData.scoreThreshold}
                    onValueChange={(value) => setFormData({ ...formData, scoreThreshold: value })}
                    max={1}
                    min={0}
                    step={0.1}
                    className="w-full"
                  />
                </div>
              )}

              {formData.retrievalMode === "hybrid-weight" && (
                <div className="space-y-2">
                  <Label>
                    语义权重: {formData.hybridWeight[0]} / 关键词权重: {1 - formData.hybridWeight[0]}
                  </Label>
                  <Slider
                    value={formData.hybridWeight}
                    onValueChange={(value) => setFormData({ ...formData, hybridWeight: value })}
                    max={1}
                    min={0}
                    step={0.1}
                    className="w-full"
                  />
                </div>
              )}

              {formData.retrievalMode === "hybrid-rerank" && (
                <div className="space-y-2">
                  <Label>Rerank 模型</Label>
                  <Select
                    value={formData.rerankModel}
                    onValueChange={(value) => setFormData({ ...formData, rerankModel: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="选择 Rerank 模型" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="bge-reranker-large">bge-reranker-large</SelectItem>
                      <SelectItem value="bge-reranker-base">bge-reranker-base</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>

        <DialogFooter className="gap-2">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            取消
          </Button>
          {currentStep === "basic" && (
            <Button onClick={() => setCurrentStep("index")} disabled={!formData.name}>
              下一步
            </Button>
          )}
          {currentStep === "index" && (
            <>
              <Button variant="outline" onClick={() => setCurrentStep("basic")}>
                上一步
              </Button>
              <Button onClick={() => setCurrentStep("retrieval")} disabled={!formData.embeddingModel}>
                下一步
              </Button>
            </>
          )}
          {currentStep === "retrieval" && (
            <>
              <Button variant="outline" onClick={() => setCurrentStep("index")}>
                上一步
              </Button>
              <Button onClick={handleCreate}>创建知识库</Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
