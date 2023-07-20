#pragma once

struct Layer
{
    float * weights, *res, *dif_res,* res_active,* gradient, *diff_weights,*in_data;
    int neyron_amount, size_layer;
};