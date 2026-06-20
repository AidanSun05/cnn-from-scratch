#include <cstring>
#include <iostream>
#include <vector>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "inference/nn.h"

constexpr int gridStep = 10;
constexpr int gridSize = 28;

int main() {
    nnInit();

    NNModel* model;
    if (int rc = nnLoad("model-mnist.bin", &model, 1); rc != NN_OK) {
        std::cerr << "Model load failed: " << rc << "\n";
        nnCleanup();
        return 1;
    }

    if (!glfwInit()) {
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Real-time MNIST inference", nullptr, nullptr);
    if (!window) {
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();

    std::vector<float> gridCells(gridSize * gridSize, 0);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowSize({ 300, 200 }, ImGuiCond_FirstUseEver);
        ImGui::Begin("Draw");

        ImGui::BeginChild("text", {}, 0, ImGuiWindowFlags_NoMove);

        ImVec2 c0 = ImGui::GetCursorScreenPos();
        ImDrawList* drawList = ImGui::GetWindowDrawList();

        ImVec2 mousePos = ImGui::GetMousePos();
        for (int y = 0; y < gridSize; y++) {
            float cellY0 = c0.y + y * gridStep;
            float cellY1 = c0.y + (y + 1) * gridStep;

            bool mouseInY = mousePos.y >= cellY0 && mousePos.y <= cellY1;
            for (int x = 0; x < gridSize; x++) {
                float cellX0 = c0.x + x * gridStep;
                float cellX1 = c0.x + (x + 1) * gridStep;
                float& cellVal = gridCells[y * gridSize + x];

                bool mouseInX = mousePos.x >= cellX0 && mousePos.x <= cellX1;
                if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && mouseInX && mouseInY) {
                    cellVal = 1;
                }

                if (cellVal == 1) {
                    drawList->AddRectFilled({ cellX0, cellY0 }, { cellX1, cellY1 }, IM_COL32(200, 200, 200, 40));
                }
            }
        }

        for (int x = 0; x <= gridSize * gridStep; x += gridStep) {
            drawList->AddLine({ c0.x + x, c0.y }, { c0.x + x, c0.y + gridSize * gridStep },
                IM_COL32(200, 200, 200, 40));
        }

        for (int y = 0; y <= gridSize * gridStep; y += gridStep) {
            drawList->AddLine({ c0.x, c0.y + y }, { c0.x + gridSize * gridStep, c0.y + y },
                IM_COL32(200, 200, 200, 40));
        }

        ImGui::EndChild();
        ImGui::End();

        ImGui::Begin("Clear");
        if (ImGui::Button("Clear")) {
            std::memset(gridCells.data(), 0, gridCells.size() * sizeof(float));
        }
        ImGui::End();

        ImGui::Begin("Prediction");
        const float* out = nnPredict(model, gridCells.data(), gridCells.size(), nullptr);
        for (int i = 0; i < 10; i++) {
            float pred = out[i];
            ImGui::Text("%d", i);
            ImGui::SameLine();
            ImGui::ProgressBar(pred);
        }
        ImGui::End();

        ImGui::Render();
        int displayX, displayY;
        glfwGetFramebufferSize(window, &displayX, &displayY);
        glViewport(0, 0, displayX, displayY);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    nnFree(model);
    nnCleanup();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}
