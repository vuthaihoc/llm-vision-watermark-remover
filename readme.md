# LLM Vision Watermark Remover

## Feature

- [ ] Detect watermark position
- [x] Remove watermark : using IOPaint


## API

```php
function remove_watermark($image, $x, $y, $w, $h)
    {
        // Gửi yêu cầu đến IOPaint API
        $url = 'http://localhost:5000/process_image'; 
        $response = (new Client())->post($url, [
            'json' => [
                'image' => base64_encode($image),
                'mask_x' => $x,
                'mask_y' => $y,
                'mask_width' => $w,
                'mask_height' => $h,
            ]
        ]);
        file_put_contents(base_path('out.jpg'), $response->getBody()->getContents());
    }
```