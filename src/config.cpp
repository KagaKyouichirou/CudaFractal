#include "TaskArgs.h"

#include <vector_types.h>
#include <QList>
#include <QSize>
#include <QString>

namespace ProjConf
{
// clang-format off

extern constexpr QSize DEFAULT_MAINWINDOW_SIZE { 1024, 768 };

// <dGrid, dBlock>
extern QList<std::pair<dim3, dim3>> DIMENSION_OPTIONS {
    { dim3(10, 1080), dim3(192, 1) },   // 1920 x 1080
    { dim3(10, 1440), dim3(256, 1) },   // 2560 x 1440
    { dim3(15, 2160), dim3(256, 1) },   // 3840 x 2160
};

extern constexpr TaskArgs DEFAULT_TASK {
    dim3(100, 100), dim3(8, 8),
    -0.5, 0.0,
    1.5 / 800,
    400
};

extern QString const TEXTURE_VIEW_STYLE {QStringLiteral(R"(
    min-width: 200px;
    min-height: 200px;
)")};

extern QString const INPUT_PANE_STYLE {QStringLiteral(R"(
    QScrollArea {
        min-width: 240px;
        background: transparent;
        border: none;
    };
    font-size: 16px;
)")};

// clang-format on
}  // namespace ProjConf
