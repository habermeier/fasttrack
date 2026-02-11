# Compact Form Design

## Changes Needed:

1. **Remove input fields** - no manual text entry
2. **Single line per field**:
   ```
   [Label]  [Value]  [Unit]
   [──── Slider─────────]
   ```

3. **Compact CSS**:
   - field-group: margin-bottom: 16px (was 40px)
   - ruler height: 60px (was 120px)
   - Remove input-wrapper

4. **HTML Structure**:
   ```html
   <div class="field-group">
       <div class="field-header">
           <span class="field-label">Glucose</span>
           <span class="field-value" id="value-glucose">90 <span class="unit">mg/dL</span></span>
       </div>
       <div class="ruler-picker" id="ruler-glucose">
           <div class="ruler-marker"></div>
           <div class="ruler-container"></div>
       </div>
   </div>
   ```

5. **Hidden Inputs**: Keep input elements hidden for form submission, but don't display them

6. **Update ruler callback**: Update the displayed value when slider moves

This will reduce height by ~60% and make it pure slider-based!
